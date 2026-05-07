#!/bin/bash
# 在 N46H2 集群安装 TransformerEngine（启用 NVSHMEM）
# 使用前提：已通过 module 加载环境，并进入 megatron310 conda
# 路径：scripts/install_te_n46h2.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# ------ NVSHMEM 配置 ------
# NVSHMEM 头文件/库在 nvhpc/25.11 安装路径下，不需要 module load
# (module load nvhpc/25.11 会替换系统 nvcc，导致 TE 编译失败)
export NVSHMEM_HOME=/data/apps/nvhpc/25.11/Linux_x86_64/25.11/comm_libs/12.9/nvshmem
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$NVSHMEM_HOME/lib/python:${PYTHONPATH:-}
export NVTE_ENABLE_NVSHMEM=1

# ------ CUDA 架构 ------
# gpu_a800: SM80; gpu_h100/h200: SM90
# 注意：TE 用的环境变量是 NVTE_CUDA_ARCHS，不是 NVTE_CUDA_ARCHITECTURES
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export NVTE_CUDA_ARCHS="80;90"

# 强制 CMake 使用系统 CUDA 12.8 编译器，不要用 NVHPC 自带的
# (TE 支持通过 NVTE_CMAKE_EXTRA_ARGS 传额外 CMake 参数，见 setup.py:85)
module load cudnn/9.6.0.74_cuda12
export CUDA_HOME=/data/apps/cuda/12.8
export CUDNN_HOME=/data/apps/cudnn/9.6.0.74_cuda12
export NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCMAKE_CXX_COMPILER=$(which g++) -DNVTE_BUILD_THIRDPARTY_CUDNN=OFF -DCUDNN_INCLUDE_DIR=$CUDNN_HOME/include -DCUDNN_LIBRARY=$CUDNN_HOME/lib/libcudnn.so"

# ------ 安装 ------
cd third_party/TransformerEngine

NVTE_ENABLE_NVSHMEM=1 \
    NVSHMEM_HOME="$NVSHMEM_HOME" \
    NVTE_FRAMEWORK=pytorch \
    pip install --no-deps --no-build-isolation -e . -vvv

cd ../..
echo "TransformerEngine installed with NVSHMEM support"
