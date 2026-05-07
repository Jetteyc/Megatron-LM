#!/bin/bash
# N46H2 shared setup for Megatron-LM jobs.
# Usage in shell: source scripts/n46h2_env.sh

module purge
module load miniforge3/25.11.0-1
module load cuda/12.8
module load cudnn/9.6.0.74_cuda12
module load gcc/12.4.0

# Keep conda artifacts out of tiny HOME quota.
export CONDA_PKGS_DIRS=/data/home/scyb091/run/.conda/pkgs
export CONDA_ENVS_PATH=/data/home/scyb091/run/.conda/envs

source activate megatron310

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1

# NVSHMEM runtime library (needed by TransformerEngine with NVSHMEM support)
# Must be after conda activate, since conda overwrites LD_LIBRARY_PATH
export NVSHMEM_HOME=/data/apps/nvhpc/25.11/Linux_x86_64/25.11/comm_libs/12.9/nvshmem
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
