#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please activate the conda env first (e.g. conda activate janusmesh)."
  exit 1
fi

echo "[1/5] Configure CUDA/GCC toolchain"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "[2/5] Install CUDA-adjacent runtime deps"
python -m pip install --upgrade pip
python -m pip install cupy-cuda11x
python -m pip install "kaolin==0.17.0" -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html

echo "[3/5] Install source-built rendering extensions"
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8"
python -m pip install --no-build-isolation "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d"
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@b6a77ad7aaf41ed90fca80ce6a2bac3c462a7881"

echo "[4/5] Install flash-attn (can take a while)"
python -m pip install --no-build-isolation flash-attn==2.8.3

echo "[5/5] Validate critical imports"
python - <<'PY'
import torch
import cupy
import nvdiffrast.torch as dr
import pytorch3d
from diff_gaussian_rasterization import GaussianRasterizationSettings
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cupy", cupy.__version__)
print("nvdiffrast ok")
print("pytorch3d", pytorch3d.__version__)
print("diff_gaussian_rasterization ok")
PY

echo "Extension setup completed."
