#!/bin/bash
# 在 N46H2 集群安装 DeepEP（仅 H100/H200 SM90）
# DeepEP v2 使用 NCCL Gin 后端，需要 SM90 + NCCL 2.30.4+
# 路径：scripts/install_deepep_n46h2.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# 只支持 SM90 (H100/H200)
export TORCH_CUDA_ARCH_LIST="9.0"

# NCCL (通过 pip wheel 安装 nvidia-nccl-cu13>=2.30.4)
NCCL_PIP_DIR=/data/home/scyb091/run/.conda/envs/megatron310/lib/python3.10/site-packages/nvidia/nccl
ln -sf libnccl.so.2 "$NCCL_PIP_DIR/lib/libnccl.so"
export EP_NCCL_ROOT_DIR="$NCCL_PIP_DIR"

# NVSHMEM
NVSHMEM_DIR=/data/apps/nvhpc/25.11/Linux_x86_64/25.11/comm_libs/12.9/nvshmem
export EP_NVSHMEM_ROOT_DIR="$NVSHMEM_DIR"

# Linker search paths (all three needed for the build)
export LIBRARY_PATH=/data/apps/cuda/12.8/lib64/stubs:$NCCL_PIP_DIR/lib:$NVSHMEM_DIR/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$NCCL_PIP_DIR/lib:$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}

# JIT cache
export EP_JIT_CACHE_DIR=/data/home/scyb091/run/.deep_ep_cache
mkdir -p "$EP_JIT_CACHE_DIR"

# 使用系统 nvcc（CUDA 12.8），不用 NVHPC 自带的 nvcc
export CUDA_HOME=/data/apps/cuda/12.8

cd third_party/DeepEP
rm -rf build

echo "=== Building DeepEP ==="
python setup.py build 2>&1

echo ""
echo "=== Installing DeepEP ==="
python setup.py install 2>&1

cd ../..
echo "DeepEP installed successfully"
