"""Capture post-RoPE keys/values/queries of every full-attention layer for one context.

Prefill a context of Qdrant Rust source (signatures only, via rust_strip), then answer a few
questions greedily with full attention, recording each decode step's queries. Output:
<out>/ctx.safetensors with, per full layer L:
  k_L, v_L      [n_kv, N, d]      prefill keys/values (post-RoPE keys)
  qp_L          [n_q, N, d]       prefill queries (post-RoPE)
  qd_L          [n_q, T, d]       decode queries, all questions concatenated
and meta.json (layers, N, per-question step counts, answers).

CPU only; meant for a small Qwen3.5 (same hybrid layout as the 9B the repo uses).
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "rust_strip", Path(__file__).resolve().parents[2] / "kv_search/data/rust_strip.py"
)
_rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rs)
strip_file = _rs.strip_file

REC: dict[int, list[torch.Tensor]] = {}
_cur_layer: list[int] = [-1]
_orig_rope = mq.apply_rotary_pos_emb
_orig_fwd = mq.Qwen3_5Attention.forward


def _rope(q, k, cos, sin, *a, **kw):
    q2, k2 = _orig_rope(q, k, cos, sin, *a, **kw)
    REC.setdefault(_cur_layer[0], []).append(q2[0].float().clone())  # [n_q, t, d]
    return q2, k2


def _fwd(self, *a, **kw):
    _cur_layer[0] = self.layer_idx
    return _orig_fwd(self, *a, **kw)


mq.apply_rotary_pos_emb = _rope
mq.Qwen3_5Attention.forward = _fwd


def build_context(root: Path, dirs: list[str], tok, budget: int) -> str:
    parts, n = [], 0
    for d in dirs:
        for f in sorted((root / d).rglob("*.rs")):
            if "test" in f.name or "/tests/" in str(f) or "benches" in str(f):
                continue
            try:
                src = strip_file(f.read_text())
            except Exception:
                continue
            block = f"// ===== {f.relative_to(root)} =====\n{src}\n"
            t = len(tok.encode(block, add_special_tokens=False))
            if n + t > budget:
                return "".join(parts)
            parts.append(block)
            n += t
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--qdrant", default="/home/user/qdrant")
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--questions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]

    text = build_context(Path(args.qdrant), args.dirs, tok, args.tokens - 64)
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": "Here is part of a Rust codebase:\n\n" + text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    ids = tok(prefix, return_tensors="pt").input_ids
    N = ids.shape[1]
    print(f"context tokens: {N}", flush=True)

    t0 = time.time()
    cache = None
    with torch.no_grad():
        for s in range(0, N, 2048):
            out = model(input_ids=ids[:, s : s + 2048], past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            print(f"  prefill {min(s + 2048, N)}/{N}  {time.time() - t0:.0f}s", flush=True)
    tensors: dict[str, torch.Tensor] = {}
    for L in full:
        tensors[f"qp_{L}"] = torch.cat(REC[L], dim=1).half()
        k = cache.layers[L].keys[0] if hasattr(cache, "layers") else cache.key_cache[L][0]
        v = cache.layers[L].values[0] if hasattr(cache, "layers") else cache.value_cache[L][0]
        tensors[f"k_{L}"] = k.float().clone()
        tensors[f"v_{L}"] = v.float().half()
    REC.clear()

    meta = {"layers": full, "N": N, "steps": [], "answers": [], "questions": args.questions, "dirs": args.dirs}
    dec: dict[int, list[torch.Tensor]] = {L: [] for L in full}
    for q in args.questions:
        qtxt = tok.apply_chat_template(
            [{"role": "user", "content": "Here is part of a Rust codebase:\n\n" + text},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        assert qtxt.startswith(prefix), "chat template prefix mismatch"
        q_ids = tok(qtxt[len(prefix):], return_tensors="pt", add_special_tokens=False).input_ids
        c = copy.deepcopy(cache)
        gen = []
        with torch.no_grad():
            out = model(input_ids=q_ids, past_key_values=c, use_cache=True, logits_to_keep=1)
            REC.clear()  # drop the question-pass queries; keep decode steps only
            for _ in range(args.gen):
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                gen.append(int(nxt))
                if gen[-1] == tok.eos_token_id:
                    break
                out = model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True)
        for L in full:
            dec[L].append(torch.cat(REC[L], dim=1))
        REC.clear()
        ans = tok.decode(gen, skip_special_tokens=True)
        meta["steps"].append(dec[full[0]][-1].shape[1])
        meta["answers"].append(ans)
        print(f"Q: {q}\nA: {ans!r}  ({time.time() - t0:.0f}s)", flush=True)
    for L in full:
        tensors[f"qd_{L}"] = torch.cat(dec[L], dim=1).float()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, out_dir / "ctx.safetensors")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print("saved", out_dir, flush=True)


if __name__ == "__main__":
    main()
