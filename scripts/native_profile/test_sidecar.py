"""Check the values-sidecar decode path against the query_batch path, through the real extension.

Uses the 4 shards native_profile built (shard0..3) as the 4 KV heads of every one of the 8
full-attention layers (hard links, own WAL each), replays recorded decode queries through
NativeEdgeRetriever.retrieve before and after export_values, and compares outputs and time.

usage: python test_sidecar.py <native_module_dir> <shards_dir> <profile_data_dir> [steps]
"""

import gc
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
from kv_search._native import NativeEdgeRetriever  # noqa: E402

shards_dir, data_dir = Path(sys.argv[2]), Path(sys.argv[3])
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 100
n, d, total, n_shards = map(int, (data_dir / "meta.txt").read_text().split())
q = [np.fromfile(data_dir / f"shard{i}_queries.f32", dtype=np.float32).reshape(total, 4, d)
     for i in range(n_shards)]

root = Path(tempfile.mkdtemp(dir=shards_dir))  # same filesystem, for hard links
try:
    layers = list(range(3, 32, 4))
    spec = []
    for L in layers:
        for h in range(4):
            p = root / f"layer{L:02d}_head{h}"
            # hard-link the shard (segments, graph) but give each copy its own WAL: edge locks it
            src_dir = (shards_dir / f"shard{h}").resolve()
            shutil.copytree(src_dir, p, copy_function=os.link, ignore=shutil.ignore_patterns("wal", "values.f32*"))
            shutil.copytree(src_dir / "wal", p / "wal")
            spec.append(((L, h), str(p)))

    scaling = 1 / np.sqrt(d)


    def step_query(t):  # [16 q-heads, 1, d]: KV head h owns q-heads 4h..4h+3
        return np.ascontiguousarray(np.stack([q[h][t] for h in range(4)]).reshape(16, 1, d))


    def run(eng, label):
        outs, ms = [], []
        for t in range(steps):
            qt = step_query(t)
            t0 = time.perf_counter()
            for L in layers:  # one token: 8 layers, serial as in decode
                o = eng.retrieve(L, qt, 128, scaling)
            ms.append((time.perf_counter() - t0) * 1e3)
            outs.append(o)
        ms = np.array(ms[5:])
        print(f"{label:<28} per token (8 layers): median {np.median(ms):6.1f} ms  p90 {np.percentile(ms, 90):6.1f}")
        return outs


    old = NativeEdgeRetriever(spec, exact=False)
    assert old.sidecar_shards == 0
    out_old = run(old, "query_batch + values (today)")

    t0 = time.perf_counter()
    written = old.export_values(spec)
    print(f"export_values: {written} shards in {time.perf_counter() - t0:.1f}s")
    del old  # releases the shards' WAL locks; the new engine loads the sidecars from disk
    gc.collect()
    new = NativeEdgeRetriever(spec, exact=False)
    assert new.sidecar_shards == 32, new.sidecar_shards
    out_new = run(new, "sidecar, values by offset")

    dmax = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(out_old, out_new))
    lmax = max(float(np.abs(a[1] - b[1]).max()) for a, b in zip(out_old, out_new))
    bmax = max(float(np.abs(a[2] - b[2]).max()) for a, b in zip(out_old, out_new))
    print(f"shards loaded with sidecar: {new.sidecar_shards}")
    print(f"max |out diff| {dmax:.2e}  |lse diff| {lmax:.2e}  |boundary diff| {bmax:.2e}")
    del new
    gc.collect()
finally:
    gc.collect()
    shutil.rmtree(root)  # 32 WAL copies + sidecars: ~12 GB at 128K
