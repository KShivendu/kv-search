# 06 · Does the traversal fetch (K) shrink with context?

> **Proxy run.** Qwen3.5-0.8B on CPU, not the 9B shards of §05. Same hybrid layout (a full-attention
> layer every 4th layer; here 6 layers × 2 KV heads = 12 shards, head dim 256). 128K pending.

§05 left one question open: K was 55.9% of context at 100K, and it decides whether edge is viable.
If K% falls fast with context, edge works at 1M. If it stays flat, edge downloads most of the cache.

## Setup

- **Context:** Qdrant Rust source (signatures via `rust_strip`), one prefill to 128K, forked at
  16K / 32K / 64K / 128K. Keys at a checkpoint are exactly the first n keys of the long prefill.
- **Queries:** 3 questions, each answered greedily for 256 tokens with full attention. Every
  question token and generated token is replayed, for all 4 query heads of a shard (as in §05).
- **Graph:** faiss `IndexHNSWFlat`, inner product, M=16 (32 links on level 0), ef_construct=100,
  grown incrementally across checkpoints.
- **Search:** a Python copy of Qdrant's `graph_layers.rs` (greedy `search_entry` on upper levels,
  `search_on_level` beam on level 0, ef = max(ef, top)), logging every node scored. It returns the
  same top-128 as faiss's own search (overlap 1.000, recall 0.948 at 32K).
- **K%** = distinct keys scored over the whole question + generation, per shard, / context.
  **V%** = distinct keys returned in any top-128 / context.

Scripts: `scripts/zha_bench/capture_long.py`, `scripts/zha_bench/measure_k.py`.

## Result (ef = 128, k = 128)

| context | K% | V% | K/V | recall@128 | nodes scored / query | K% after the question only |
|---|---|---|---|---|---|---|
| 16K | 85.2 | 38.4 | 2.2 | 0.921 | 1,338 | 58.5 |
| 32K | 72.4 | 27.3 | 2.6 | 0.903 | 1,478 | 42.7 |
| 64K | 63.9 | 20.1 | 3.2 | 0.867 | 1,706 | 30.5 |

Per layer at 64K, K% spans 60.2 (layer 11) to 71.0 (layer 7). No layer escapes.

## Reading

- **K% falls, but slowly: about context^−0.21.** Fitted on these three points it predicts 58% at
  100K; §05 measured 55.9% at 100K on the 9B. A different model and graph builder land on the same
  number, so the trend looks real.
- **Extrapolation (3 points, log-log):** ~50% at 200K, ~36% at 1M. V% falls faster
  (~context^−0.47), ~6% at 1M.
- **Still climbing at the end of generation:** +3.0 points of K% in the last 64 tokens at 32K and
  64K. Longer answers fetch more.
- **Per-query work grows with context** (1,338 → 1,706 nodes scored) while recall drops
  (0.92 → 0.87). Holding recall fixed needs a larger ef, which means more traversal.

In bytes for Qwen3.5-9B at 1M (32 KB/token of K+V, so 16 GB keys + 16 GB values), the extrapolation
gives ~5.8 GB of keys scored plus ~0.9 GB of values per 256-token answer: about **1/5 of the cache**.
Edge is not dead at 1M on this trend, but it does not reach "a few percent" either. Almost all of
the cost is traversal, so the levers are the ones that cut what the walk touches: a better-conditioned
graph (whitened or carrier-removed keys, exact for ranking) so a smaller ef keeps recall, traversal
biased toward already-cached nodes, and shipping the upper levels with edge.
