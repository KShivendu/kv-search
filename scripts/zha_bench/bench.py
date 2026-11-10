"""Offline bench: HNSW top-k vs ZHA-style chunk routing over one captured context.

For every (full-attention layer, KV head) shard, each decode query (4 GQA query heads per shard)
picks a set S of prefill keys; we compare attention over S with exact attention over all prefill
keys:
  mass    = softmax mass of S under exact attention
  rel_err = ||out_S - out|| / ||out||   (out = softmax(qK^T/sqrt d) V)

Methods (budget B = keys attended):
  exact        exact top-B (ceiling for any top-B method)
  hnsw_efX     hnswlib inner-product HNSW, M=16, ef_construct=100, ef=max(B, X*B)
  chunk_*      contiguous chunks of C keys; score chunks, attend all keys of the top B/C chunks
     oracle    true chunk log-mass (ceiling for chunk routing)
     centroid  q . mean(K_c)/sqrt d (+log C): the tangent plane of the chunk log-mass at q = 0
     zhaR_fit  ZHA summaries: max_r of tangent planes at R learned reference queries, refs fitted
               by KL to the pooled chunk distribution. fit = indoc (prefill queries of this
               context, causal) or xdoc (queries + keys of another context)
  rerank       zha16 picks chunks covering 4B keys, then exact top-B inside them
  zha top-p    adaptive: smallest chunk set whose estimated mass >= p
"""

import argparse
import json
import math
import time
from pathlib import Path

import hnswlib
import numpy as np
import torch
from safetensors.torch import load_file

torch.set_num_threads(4)
G = 4  # query heads per KV head (Qwen3.5-0.8B: 8 q heads, 2 kv heads)


def load(d: Path):
    meta = json.loads((d / "meta.json").read_text())
    return meta, load_file(d / "ctx.safetensors")


# ---------- ZHA summaries ----------

def summaries(refs: torch.Tensor, K: torch.Tensor, C: int, scale: float):
    """refs [R,d], K [N,d] -> khat [R,nC,d], b [R,nC]: tangent planes of log sum_j exp(q.k_j*scale)
    at q = r, per chunk.  t_r(q) = q.khat*scale + b."""
    nC = K.shape[0] // C
    Kc = K[: nC * C].view(nC, C, -1)
    s = torch.einsum("rd,ncd->rnc", refs, Kc) * scale  # [R,nC,C]
    a = torch.softmax(s, dim=-1)
    khat = torch.einsum("rnc,ncd->rnd", a, Kc)
    lse = torch.logsumexp(s, dim=-1)
    b = lse - torch.einsum("rd,rnd->rn", refs, khat) * scale
    return khat, b


def zha_scores(q: torch.Tensor, khat: torch.Tensor, b: torch.Tensor, scale: float):
    """q [Q,d] -> [Q,nC] estimated chunk log-mass (max over reference tangents)."""
    t = torch.einsum("qd,rnd->qrn", q, khat) * scale + b[None]
    return t.max(dim=1).values


def chunk_lse(q: torch.Tensor, K: torch.Tensor, C: int, scale: float):
    nC = K.shape[0] // C
    s = (q @ K[: nC * C].T) * scale
    return torch.logsumexp(s.view(q.shape[0], nC, C), dim=-1)


def fit_refs(train, R: int, C: int, scale: float, steps: int = 300, seed: int = 0):
    """train: list of (q [Q,d], K [N,d], vis [Q] number of visible chunks or None).
    Learn R reference queries (shared across chunks) by KL(true chunk dist || predicted)."""
    g = torch.Generator().manual_seed(seed)
    allq = torch.cat([t[0] for t in train])
    norm0 = allq.norm(dim=-1).mean()
    init = allq[torch.randperm(allq.shape[0], generator=g)[:R]] / norm0
    w = torch.nn.Parameter(init.clone())
    opt = torch.optim.Adam([w], lr=2e-2)
    targets = []
    for q, K, vis in train:
        lse = chunk_lse(q, K, C, scale)
        mask = None
        if vis is not None:
            mask = torch.arange(lse.shape[1])[None] >= vis[:, None]
            lse = lse.masked_fill(mask, -1e30)
        targets.append((torch.log_softmax(lse, -1), mask))
    for it in range(steps):
        opt.zero_grad()
        loss = 0.0
        for (q, K, _), (tgt, mask) in zip(train, targets):
            khat, b = summaries(w * norm0, K, C, scale)
            pred = zha_scores(q, khat, b, scale)
            if mask is not None:
                pred = pred.masked_fill(mask, -1e30)
            lp = torch.log_softmax(pred, -1)
            p = tgt.exp()
            loss = loss + (p * (tgt - lp)).sum(-1).mean()
        loss = loss / len(train)
        loss.backward()
        opt.step()
    return (w * norm0).detach(), float(loss)


# ---------- evaluation ----------

_EXACT: dict = {}


def eval_sets(q, K, V, scale, sel_fn):
    """sel_fn(q_i_index) -> LongTensor of key ids. Returns per-query mass, rel_err, n_keys."""
    key = (id(q), id(K))
    if key not in _EXACT:
        _EXACT.clear()
        p_ = torch.softmax((q @ K.T) * scale, -1)
        _EXACT[key] = (p_, p_ @ V, q, K)  # q, K kept alive so ids are not reused
    p, out = _EXACT[key][:2]
    mass, err, nk = [], [], []
    for i in range(q.shape[0]):
        S = sel_fn(i)
        ps = p[i, S]
        m = ps.sum()
        o = (ps / m) @ V[S]
        mass.append(float(m))
        err.append(float((o - out[i]).norm() / out[i].norm()))
        nk.append(len(S))
    return np.array(mass), np.array(err), np.array(nk)


def chunk_ids(chunks: torch.Tensor, C: int):
    return (chunks[:, None] * C + torch.arange(C)[None]).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", required=True, help="eval context dir (capture.py output)")
    ap.add_argument("--xdoc", default=None, help="other context dir for cross-doc ref fitting")
    ap.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--chunks", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--refs", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--topp", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.9, 0.95])
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta, T = load(Path(args.ctx))
    xT = load(Path(args.xdoc))[1] if args.xdoc else None
    layers = meta["layers"]
    rows: list[dict] = []
    fit_log: list[dict] = []
    t0 = time.time()

    def add(layer, head, method, C, R, budget, m, e, nk):
        for i in range(len(m)):
            rows.append(dict(layer=layer, head=head, method=method, C=C, R=R, budget=budget,
                             mass=m[i], err=e[i], keys=int(nk[i])))

    for L in layers:
        K_all, V_all = T[f"k_{L}"].float(), T[f"v_{L}"].float()
        d = K_all.shape[-1]
        scale = 1 / math.sqrt(d)
        for h in range(K_all.shape[0]):
            K, V = K_all[h], V_all[h]
            N = K.shape[0]
            q = T[f"qd_{L}"][h * G:(h + 1) * G].float().reshape(-1, d)  # [G*steps, d]
            s_exact = (q @ K.T) * scale
            order = s_exact.argsort(dim=-1, descending=True)

            for B in args.budgets:
                add(L, h, "exact", 0, 0, B, *eval_sets(q, K, V, scale, lambda i: order[i, :B]))

            # HNSW
            idx = hnswlib.Index(space="ip", dim=d)
            idx.init_index(max_elements=N, M=16, ef_construction=100, random_seed=0)
            idx.add_items(K.numpy(), np.arange(N), num_threads=4)
            for mult in (1, 4):
                for B in args.budgets:
                    idx.set_ef(max(B, mult * B))
                    lab, _ = idx.knn_query(q.numpy(), k=B, num_threads=4)
                    lab_t = torch.from_numpy(lab.astype(np.int64))
                    add(L, h, f"hnsw_ef{mult}x", 0, 0, B,
                        *eval_sets(q, K, V, scale, lambda i: lab_t[i]))

            # training sets for refs
            qp = T[f"qp_{L}"][h * G:(h + 1) * G].float()  # [G, N, d]
            gen = torch.Generator().manual_seed(L * 10 + h)
            pos = torch.randint(N // 4, N, (args.n_train,), generator=gen)
            hd = torch.randint(0, G, (args.n_train,), generator=gen)
            q_in = qp[hd, pos]
            train_in = {C: [(q_in, K, (pos + 1) // C)] for C in args.chunks}
            train_x = None
            if xT is not None:
                Kx = xT[f"k_{L}"][h].float()
                qx_d = xT[f"qd_{L}"][h * G:(h + 1) * G].float().reshape(-1, d)
                qx_p = xT[f"qp_{L}"][h * G:(h + 1) * G].float()
                Nx = Kx.shape[0]
                posx = torch.randint(Nx // 4, Nx, (args.n_train,), generator=gen)
                hdx = torch.randint(0, G, (args.n_train,), generator=gen)
                train_x = {C: [(qx_p[hdx, posx], Kx, (posx + 1) // C), (qx_d, Kx, None)]
                           for C in args.chunks}

            for C in args.chunks:
                nC = N // C
                Kc_ids_valid = nC * C
                true_lse = chunk_lse(q, K, C, scale)
                scorers = {"oracle": (true_lse, 0), "centroid": None}
                mean_k = K[:Kc_ids_valid].view(nC, C, d).mean(1)
                scorers["centroid"] = ((q @ mean_k.T) * scale + math.log(C), 1)
                for R in args.refs:
                    refs, loss = fit_refs(train_in[C], R, C, scale, steps=args.steps)
                    kh, b = summaries(refs, K, C, scale)
                    scorers[f"zha_indoc"] = scorers.get("zha_indoc", {})
                    scorers[f"zha{R}_indoc"] = (zha_scores(q, kh, b, scale), R)
                    fit_log.append(dict(layer=L, head=h, C=C, R=R, fit="indoc", kl=loss))
                    if train_x is not None and R == max(args.refs):
                        refs_x, loss_x = fit_refs(train_x[C], R, C, scale, steps=args.steps)
                        khx, bx = summaries(refs_x, K, C, scale)
                        scorers[f"zha{R}_xdoc"] = (zha_scores(q, khx, bx, scale), R)
                        fit_log.append(dict(layer=L, head=h, C=C, R=R, fit="xdoc", kl=loss_x))
                scorers.pop("zha_indoc", None)

                for name, (sc, R) in scorers.items():
                    corder = sc.argsort(dim=-1, descending=True)
                    for B in args.budgets:
                        nsel = max(1, B // C)
                        add(L, h, f"chunk_{name}", C, R, B, *eval_sets(
                            q, K, V, scale, lambda i: chunk_ids(corder[i, :nsel], C)))
                    if name == f"zha{max(args.refs)}_indoc":
                        # rerank: chunks covering 4B keys, exact top-B inside
                        for B in args.budgets:
                            nsel = max(1, 4 * B // C)

                            def rr(i, nsel=nsel, B=B):
                                cand = chunk_ids(corder[i, :nsel], C)
                                top = s_exact[i, cand].argsort(descending=True)[:B]
                                return cand[top]
                            add(L, h, "rerank_zha16x4", C, R, B, *eval_sets(q, K, V, scale, rr))
                        # top-p on estimated chunk distribution
                        est = torch.softmax(sc, -1)
                        srt, ordp = est.sort(dim=-1, descending=True)
                        cum = srt.cumsum(-1)
                        for p in args.topp:
                            n_need = (cum < p).sum(-1) + 1
                            add(L, h, f"topp_zha16", C, R, p, *eval_sets(
                                q, K, V, scale, lambda i: chunk_ids(ordp[i, :n_need[i]], C)))
                    if name == "oracle":
                        est = torch.softmax(sc, -1)
                        srt, ordp = est.sort(dim=-1, descending=True)
                        cum = srt.cumsum(-1)
                        for p in args.topp:
                            n_need = (cum < p).sum(-1) + 1
                            add(L, h, f"topp_oracle", C, 0, p, *eval_sets(
                                q, K, V, scale, lambda i: chunk_ids(ordp[i, :n_need[i]], C)))
            print(f"layer {L} head {h} done  {time.time() - t0:.0f}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd  # noqa: PLC0415
    df = pd.DataFrame(rows)
    df.to_parquet(out / "rows.parquet")
    (out / "fit_log.json").write_text(json.dumps(fit_log, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
