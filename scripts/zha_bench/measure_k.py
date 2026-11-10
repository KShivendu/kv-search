"""Traversal fetch (K) vs context length, from capture_long.py output.

Per (full-attention layer, KV head) shard and checkpoint length n: build an inner-product HNSW over
the first n keys (faiss, M=16 -> 32 links on level 0, ef_construct=100, grown incrementally across
checkpoints), then replay every recorded query (question tokens + generated tokens, 4 GQA query
heads) through a Python copy of Qdrant's search (graph_layers.rs: greedy search_entry on upper
levels, search_on_level beam on level 0 with ef = max(ef, top)), logging every node scored.

Per shard and question:
  K%  distinct nodes scored over the whole question+generation / n   (traversal fetch)
  V%  distinct nodes returned in any top-k / n                       (value fetch)
  recall@k vs exact top-k
Output: <dir>/k_results.json and a printed table.
"""

import argparse
import heapq
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import faiss
import numpy as np
from safetensors.numpy import load_file

G = 4


class Graph:
    def __init__(self, index: faiss.IndexHNSWFlat, X: np.ndarray):
        h = index.hnsw
        self.X = X
        self.n = index.ntotal
        nb = faiss.vector_to_array(h.neighbors)
        off = faiss.vector_to_array(h.offsets).astype(np.int64)
        cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
        self.levels = faiss.vector_to_array(h.levels)[: self.n] - 1  # top level of each node
        self.entry = int(h.entry_point)
        self.max_level = int(h.max_level)
        w0 = int(cum[1])
        self.l0 = nb[off[: self.n, None] + np.arange(w0)[None]]  # [n, 32], -1 padded
        self.up: dict[int, dict[int, np.ndarray]] = {}
        for lvl in range(1, self.max_level + 1):
            a, b = int(cum[lvl]), int(cum[lvl + 1])
            ids = np.nonzero(self.levels >= lvl)[0]
            d = {}
            for i in ids:
                row = nb[off[i] + a: off[i] + b]
                d[int(i)] = row[row >= 0]
            self.up[lvl] = d
        self.stamp = np.zeros(self.n, dtype=np.int64)
        self.cur = 0

    def search(self, q: np.ndarray, k: int, ef: int, visited_out: set):
        X = self.X
        ef = max(ef, k)
        ep = self.entry
        s_ep = float(X[ep] @ q)
        visited_out.add(ep)
        for lvl in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                links = self.up[lvl].get(ep)
                if links is None or len(links) == 0:
                    break
                sc = X[links] @ q
                visited_out.update(links.tolist())
                j = int(sc.argmax())
                if sc[j] > s_ep:
                    ep, s_ep, changed = int(links[j]), float(sc[j]), True
        self.cur += 1
        st = self.stamp
        st[ep] = self.cur
        nearest = [(s_ep, ep)]  # min-heap, size <= ef
        cand = [(-s_ep, ep)]  # max-heap
        while cand:
            negs, c = heapq.heappop(cand)
            if len(nearest) >= ef and -negs < nearest[0][0]:
                break
            links = self.l0[c]
            links = links[links >= 0]
            links = links[st[links] != self.cur]
            if len(links) == 0:
                continue
            st[links] = self.cur
            visited_out.update(links.tolist())
            sc = X[links] @ q
            for s, p in zip(sc.tolist(), links.tolist()):
                if len(nearest) < ef:
                    heapq.heappush(nearest, (s, p))
                    heapq.heappush(cand, (-s, p))
                elif s > nearest[0][0]:
                    heapq.heapreplace(nearest, (s, p))
                    heapq.heappush(cand, (-s, p))
        top = heapq.nlargest(k, nearest)
        return np.array([p for _, p in top], dtype=np.int64)


def run_shard(job):
    d, L, h, lengths, n_q, k, efs = job
    K_all = load_file(str(Path(d) / "keys.safetensors"))[f"k_{L}"][h].astype(np.float32)
    dim = K_all.shape[1]
    index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 100
    faiss.omp_set_num_threads(1)
    out = []
    built = 0
    for n in lengths:
        index.add(K_all[built:n])
        built = n
        g = Graph(index, K_all[:n])
        qs = load_file(str(Path(d) / f"q_{n}.safetensors"))
        for j in range(n_q):
            Q = qs[f"q_{L}_{j}"][h * G:(h + 1) * G].astype(np.float32)  # [G, T, d]
            T = Q.shape[1]
            exact = np.argpartition(-(Q.reshape(-1, dim) @ K_all[:n].T), k, axis=1)[:, :k]
            exact = exact.reshape(G, T, k)
            for ef in efs:
                vis: set = set()
                ret: set = set()
                curve = []
                rec = []
                per_q = []
                for t in range(T):
                    for gq in range(G):
                        before = len(vis)
                        one: set = set()
                        ids = g.search(Q[gq, t], k, ef, one)
                        per_q.append(len(one))
                        vis |= one
                        ret.update(ids.tolist())
                        rec.append(len(set(ids.tolist()) & set(exact[gq, t].tolist())) / k)
                    curve.append(len(vis) / n)
                out.append(dict(n=n, layer=L, head=h, q=j, ef=ef, T=T,
                                K=len(vis) / n, V=len(ret) / n, recall=float(np.mean(rec)),
                                visited_per_query=float(np.mean(per_q)), curve=curve))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--ef", type=int, nargs="+", default=[128])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lengths", type=int, nargs="*", default=None)
    args = ap.parse_args()
    meta = json.loads((Path(args.dir) / "meta.json").read_text())
    lengths = args.lengths or meta["lengths"]
    n_q = len(meta["questions"])
    jobs = [(args.dir, L, h, lengths, n_q, args.k, args.ef) for L in meta["layers"] for h in range(2)]
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(run_shard, jobs):
            res += r
            print(f"shard L{r[0]['layer']} h{r[0]['head']} done {time.time() - t0:.0f}s", flush=True)
    (Path(args.dir) / "k_results.json").write_text(json.dumps(res))
    print(f"{'n':>7} {'ef':>4} {'K%':>6} {'V%':>6} {'K/V':>5} {'recall':>6} {'vis/q':>6} {'K% after question':>18}")
    for n in lengths:
        for ef in args.ef:
            rs = [r for r in res if r["n"] == n and r["ef"] == ef]
            qtok = [meta["runs"][str(n)][r["q"]]["q_tokens"] for r in rs]
            first = np.mean([r["curve"][qt - 1] for r, qt in zip(rs, qtok)])
            K = np.mean([r["K"] for r in rs]); V = np.mean([r["V"] for r in rs])
            print(f"{n:>7} {ef:>4} {100*K:6.1f} {100*V:6.1f} {K/V:5.1f} "
                  f"{np.mean([r['recall'] for r in rs]):6.3f} {np.mean([r['visited_per_query'] for r in rs]):6.0f} "
                  f"{100*first:18.1f}")


if __name__ == "__main__":
    main()
