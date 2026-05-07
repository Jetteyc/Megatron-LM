#!/bin/bash
# 双机各 4 卡：跨节点需 --qos=gpugpu + --gres=gpu:N；勿手写 -c。
#SBATCH -J mg_2n4g_eq_runmd
#SBATCH -p gpu_a800
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --qos=gpugpu
#SBATCH -o /data/home/scyb091/run/rjy/Megatron-LM/output/slurm-%j-%x.out

set -euo pipefail
cd /data/home/scyb091/run/rjy/Megatron-LM

# module 命令在 sbatch 环境不可用，需先初始化
source /usr/share/modules/init/bash 2>/dev/null || true
source scripts/n46h2_env.sh

# N46H2 A800 跨节点 NCCL 参数
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
MASTER_PORT=${MASTER_PORT:-29531}
NNODES=${SLURM_JOB_NUM_NODES}
NPROC_PER_NODE=4

echo "nodes from slurm:"
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

srun --ntasks="${NNODES}" --ntasks-per-node=1 --cpu-bind=cores bash -lc '
  export NNODES='"${NNODES}"'
  export NODE_RANK="${SLURM_PROCID}"
  export NPROC_PER_NODE='"${NPROC_PER_NODE}"'
  export MASTER_ADDR='"${MASTER_ADDR}"'
  export MASTER_PORT='"${MASTER_PORT}"'
  export CUDA_VISIBLE_DEVICES=0,1,2,3

  tests/unit_tests/pipeline_parallel/run_test_schedules.sh 2>&1 | tee "node${SLURM_PROCID}.log"
'
