"""One long prefill, forked at growing context lengths, for the traversal-fetch (K) scaling run.

Prefill Qdrant Rust source (signatures via rust_strip) up to max(--lengths) tokens. At each
checkpoint length n the prefill cache is copied and every question is answered greedily with full
attention, recording post-RoPE queries of every full-attention layer for the question tokens and
each generated token. Keys at a checkpoint are the first n keys of the final prefill, so keys are
saved once.

Output <out>/:
  keys.safetensors           k_L [n_kv, N_max, d] (fp16)
  q_<n>.safetensors          q_L_<j> [n_q, T_j, d] for question j (question tokens + decode)
  meta.json                  layers, lengths, question token counts, answers
"""

import argparse
import copy
import ctypes
import gc
import importlib.util
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

_spec = importlib.util.spec_from_file_location(
    "rust_strip", Path(__file__).resolve().parents[2] / "kv_search/data/rust_strip.py"
)
_rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rs)

REC: dict[int, list[torch.Tensor]] = {}
RECORD = [False]
_cur = [-1]
_orig_rope = mq.apply_rotary_pos_emb
_orig_fwd = mq.Qwen3_5Attention.forward


def _rope(q, k, cos, sin, *a, **kw):
    q2, k2 = _orig_rope(q, k, cos, sin, *a, **kw)
    if RECORD[0]:
        REC.setdefault(_cur[0], []).append(q2[0].float().clone())
    return q2, k2


def _fwd(self, *a, **kw):
    _cur[0] = self.layer_idx
    return _orig_fwd(self, *a, **kw)


mq.apply_rotary_pos_emb = _rope
mq.Qwen3_5Attention.forward = _fwd


def build_context(root: Path, dirs: list[str], tok, budget: int) -> list[int]:
    ids: list[int] = []
    for d in dirs:
        for f in sorted((root / d).rglob("*.rs")):
            if "test" in f.name or "/tests/" in str(f) or "benches" in str(f):
                continue
            try:
                src = _rs.strip_file(f.read_text())
            except Exception:
                continue
            ids += tok.encode(f"// ===== {f.relative_to(root)} =====\n{src}\n", add_special_tokens=False)
            if len(ids) >= budget:
                return ids[:budget]
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--qdrant", default="/home/user/qdrant")
    ap.add_argument("--dirs", nargs="+", default=[
        "lib/collection/src", "lib/segment/src", "lib/storage/src", "src", "lib/shard/src"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--ask-at", type=int, nargs="*", default=None,
                    help="checkpoints to run questions at (default: all)")
    ap.add_argument("--questions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    cfg = model.config.get_text_config()
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]

    head = tok.apply_chat_template([{"role": "user", "content": "Here is part of a Rust codebase:\n\nX"}],
                                   tokenize=False, add_generation_prompt=False)
    pre_txt, post_txt = head.split("X", 1)
    pre = tok.encode(pre_txt, add_special_tokens=False)
    lengths = sorted(args.lengths)
    body = build_context(Path(args.qdrant), args.dirs, tok, lengths[-1] - len(pre))
    ctx_ids = torch.tensor([pre + body])
    N = ctx_ids.shape[1]
    lengths = [n for n in lengths if n <= N]
    print(f"context tokens: {N}, checkpoints {lengths}", flush=True)

    meta = {"layers": full, "lengths": lengths, "questions": args.questions, "runs": {}}
    t0 = time.time()
    cache = None
    pos = 0
    with torch.no_grad():
        for n in lengths:
            while pos < n:
                # past 64K, smaller chunks keep CPU attention scores (q x N x heads) in memory
                step = min(2048 if pos < 65536 else 512, n - pos)
                o = model(input_ids=ctx_ids[:, pos:pos + step], past_key_values=cache,
                          use_cache=True, logits_to_keep=1)
                cache = o.past_key_values
                pos += step
                print(f"  prefill {pos}/{N}  {time.time() - t0:.0f}s", flush=True)
            if n == lengths[-1] and not (out_dir / "keys.safetensors").exists():
                save_keys(cache, full, out_dir)
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            if args.ask_at is not None and n not in args.ask_at:
                continue
            qt: dict[str, torch.Tensor] = {}
            runs = []
            for j, q in enumerate(args.questions):
                if args.skip_done and (out_dir / f"q_{n}.safetensors").exists():
                    prev = load_file(out_dir / f"q_{n}.safetensors")
                    if f"q_{full[0]}_{j}" in prev:
                        qt.update({k: v for k, v in prev.items() if k.endswith(f"_{j}")})
                        runs.append(json.loads((out_dir / "meta.json").read_text())
                                    ["runs"].get(str(n), [{}] * (j + 1))[j])
                        continue
                # close the user turn mid-context, then ask
                q_txt = post_txt + tok.apply_chat_template(
                    [{"role": "user", "content": q}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                q_ids = torch.tensor([tok.encode(q_txt, add_special_tokens=False)])
                c = copy.deepcopy(cache)
                REC.clear()
                RECORD[0] = True
                o = model(input_ids=q_ids, past_key_values=c, use_cache=True, logits_to_keep=1)
                gen = []
                for _ in range(args.gen):
                    nxt = o.logits[:, -1].argmax(-1, keepdim=True)
                    gen.append(int(nxt))
                    if gen[-1] == tok.eos_token_id:
                        break
                    o = model(input_ids=nxt, past_key_values=o.past_key_values, use_cache=True)
                RECORD[0] = False
                for L in full:
                    qt[f"q_{L}_{j}"] = torch.cat(REC[L], dim=1).half().contiguous()
                REC.clear()
                del c, o  # o.past_key_values still holds this question's cache copy
                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)
                save_file(qt, out_dir / f"q_{n}.safetensors")  # partial: survives a later kill
                ans = tok.decode(gen, skip_special_tokens=True)
                runs.append({"q_tokens": q_ids.shape[1], "gen": len(gen), "answer": ans})
                print(f"[{n}] Q{j}: {len(gen)} tok  {ans[:120]!r}  ({time.time() - t0:.0f}s)", flush=True)
            save_file(qt, out_dir / f"q_{n}.safetensors")
            meta_path = out_dir / "meta.json"
            if meta_path.exists():  # keep runs recorded by an earlier, partial capture
                old = json.loads(meta_path.read_text())
                meta["runs"] = {**old.get("runs", {}), **meta["runs"]}
                meta["lengths"] = sorted(set(old.get("lengths", [])) | set(meta["lengths"]))
            meta["runs"][str(n)] = runs
            meta_path.write_text(json.dumps(meta, indent=1))
    print("done", out_dir, f"{time.time() - t0:.0f}s", flush=True)


def save_keys(cache, full: list[int], out_dir: Path) -> None:
    keys = {f"k_{L}": cache.layers[L].keys[0].half().contiguous() for L in full}
    save_file(keys, out_dir / "keys.safetensors")
    print("saved keys", flush=True)


if __name__ == "__main__":
    main()
