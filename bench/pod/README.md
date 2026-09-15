# pod — run allin1 on RunPod

allin1 does not run on this Windows box (NATTEN has no Windows wheels, no CUDA toolkit here). It also
needs NATTEN's **legacy** API (`natten2dqkrpb`/`natten2dav`, last present in NATTEN 0.15.1), whose
prebuilt wheels are no longer online — so the pod builds NATTEN from source, which needs `nvcc`.

## Pod
- Any GPU on the cheap list is fine (RTX 4000 Ada / A4000 / 3090 / L4…); it is minutes of work.
  Skip MIG slices — nothing to gain, one more thing to debug.
- **Template must be a `-devel` PyTorch image** with torch 2.0–2.2, e.g.
  `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`. (NATTEN 0.15.1 asserts torch ≥ 2.0;
  it predates torch 2.3+, so newer templates may fail to compile.)
- Disk: 20 GB is plenty (models ~1 GB, tracks ~150 MB, stems are deleted after each track).

## Steps
```bash
# local (PowerShell or Git Bash) — bundle the 14 benchmark tracks + this folder
cd camelot_from_youtube
python - <<'PY'
import json, io, tarfile, sys; sys.path.insert(0, "bench")
from beat_bench import audio_path
t = json.load(io.open("bench/tracks.json", encoding="utf-8"))
with tarfile.open("bench/pod/tracks.tgz", "w:gz") as tar:
    for v in t.values(): tar.add(audio_path(v["audio"]), arcname="tracks/" + v["audio"])
print("bench/pod/tracks.tgz")
PY
scp -P <port> bench/pod/tracks.tgz bench/pod/setup.sh bench/pod/run.sh bench/pod/run.py root@<pod-ip>:/workspace/

# on the pod
cd /workspace && tar -xzf tracks.tgz && bash setup.sh && bash run.sh /workspace/tracks

# local — fetch and score
scp -P <port> root@<pod-ip>:/workspace/allin1-results.tgz bench/
tar -xzf bench/allin1-results.tgz -C bench/          # -> bench/results/allin1/
env1/Scripts/python bench/beat_bench.py score        # allin1 row appears alongside the others
```

## What to look at
- `beat_bench.py score` → the `allin1` row on the three native-Rekordbox tracks (Flip, Waves, Quicksilver).
- `results/allin1/<track>.segments.json` → boundaries. allin1 places them on its **own downbeats**, so
  boundary timing is bounded by its downbeat accuracy; score boundaries as bar offset against the comb
  grid (`results/rigid_grid_comb/`), and compare against camelot's `events_block` in `analysis_cache.json`.
