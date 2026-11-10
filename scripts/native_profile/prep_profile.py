"""Write raw f32 inputs for native_profile from capture_long.py output.

Four shards stand in for one Qwen3.5-9B layer (4 KV heads): (layer 15, head 0/1) and
(layer 19, head 0/1) of the 0.8B capture. Values are not captured at 128K, so another layer's keys
stand in (only timing is measured; their content does not matter).

usage: python prep_profile.py <capture_dir> <out_dir> [n] [question]
"""

import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

src, out = Path(sys.argv[1]), Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 131072
qi = int(sys.argv[4]) if len(sys.argv) > 4 else 0
out.mkdir(parents=True, exist_ok=True)
cells = [(15, 0), (15, 1), (19, 0), (19, 1)]
G = 4
with safe_open(str(src / "keys.safetensors"), framework="np") as fk, \
        safe_open(str(src / f"q_{n}.safetensors"), framework="np") as fq:
    steps = None
    for i, (L, h) in enumerate(cells):
        k = fk.get_slice(f"k_{L}")[h:h + 1][0][:n].astype(np.float32)
        v = fk.get_slice("k_23")[h:h + 1][0][:n].astype(np.float32)
        q = fq.get_tensor(f"q_{L}_{qi}")[h * G:(h + 1) * G].astype(np.float32)  # [4, T, d]
        q = np.ascontiguousarray(q.transpose(1, 0, 2))  # [T, 4, d]
        steps = q.shape[0]
        k.tofile(out / f"shard{i}_keys.f32")
        v.tofile(out / f"shard{i}_values.f32")
        q.tofile(out / f"shard{i}_queries.f32")
    (out / "meta.txt").write_text(f"{n} {k.shape[1]} {steps} {len(cells)}\n")
print("wrote", out, n, steps)
