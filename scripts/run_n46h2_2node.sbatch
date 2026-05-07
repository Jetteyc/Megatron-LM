#!/bin/bash
#SBATCH -J mg_2n8g_eq_runmd
#SBATCH -p gpu_h100
#SBATCH -N 2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --qos=gpugpu
#SBATCH -o /data/home/scyb091/run/rjy/Megatron-LM/output/slurm-%j.out
set -euo pipefail
cd /data/home/scyb091/run/rjy/Megatron-LM

# module 命令在 sbatch 环境不可用，需先初始化
source /usr/share/modules/init/bash 2>/dev/null || true
source scripts/n46h2_env.sh

# N46H2 recommended NCCL settings.
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
MASTER_PORT=${MASTER_PORT:-29531}
NNODES=${SLURM_JOB_NUM_NODES}
NPROC_PER_NODE=8

echo "nodes from slurm:"
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

# Equivalent to the two-machine local run.md flow:
# each node runs run_test_schedules.sh once, and each process uses 4 GPUs.
srun --ntasks="${NNODES}" --ntasks-per-node=1 --cpu-bind=cores bash -lc '
  export NNODES='"${NNODES}"'
  export NODE_RANK="${SLURM_PROCID}"
  export NPROC_PER_NODE='"${NPROC_PER_NODE}"'
  export MASTER_ADDR='"${MASTER_ADDR}"'
  export MASTER_PORT='"${MASTER_PORT}"'
  export CUDA_VISIBLE_DEVICES=0,1,2,3

  tests/unit_tests/pipeline_parallel/run_test_schedules.sh 2>&1 | tee "node${SLURM_PROCID}.log"
'
