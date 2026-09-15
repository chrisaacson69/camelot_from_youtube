#!/usr/bin/env bash
# setup.sh - stand up allin1 on a RunPod GPU pod.
#
# Template: a RunPod *PyTorch* image with a "-devel" CUDA (has nvcc), torch 2.0-2.2 era, e.g.
#   runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
# Why that old: allin1 (1.1.0, 2024) calls NATTEN's legacy API (natten2dqkrpb / natten2dav),
# which exists up to NATTEN 0.15.1 and was removed in 0.17. No prebuilt 0.15.1 wheels are
# online any more, so it is built from source here (~5-15 min). Any card on the RunPod list
# is fine; the build needs nvcc, not a big GPU.
set -euo pipefail

echo "== environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import torch, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu ok", torch.cuda.is_available())
maj, min_ = torch.cuda.get_device_capability(0)
print("compute capability", f"{maj}.{min_}")
assert torch.cuda.is_available(), "no CUDA - wrong template"
PY
command -v nvcc >/dev/null || { echo "!! nvcc missing - use a -devel template"; exit 1; }

echo "== system deps"
apt-get update -qq && apt-get install -y -qq ffmpeg cmake ninja-build >/dev/null

echo "== NATTEN 0.15.1 from source (legacy API that allin1 needs)"
export NATTEN_CUDA_ARCH="$(python -c 'import torch; a,b=torch.cuda.get_device_capability(0); print(f"{a}.{b}")')"
export NATTEN_N_WORKERS="$(nproc)"
pip install -q "cmake==3.20.3" packaging
pip install -q --no-build-isolation "natten==0.15.1"
python -c "from natten.functional import natten1dav, natten1dqkrpb, natten2dav, natten2dqkrpb; print('natten legacy API ok')"

echo "== allin1"
pip install -q allin1
python - <<'PY'
import allin1, inspect
print("allin1", allin1.__version__ if hasattr(allin1, "__version__") else "(installed)")
print(inspect.signature(allin1.analyze))
PY
echo "== done. Next: bash run.sh /workspace/tracks"
