#!/bin/bash
# Experiment: 2 node × 4 GPU (EP=4,TP=2,CP=2)
#SBATCH -J exp_2n4g
#SBATCH -p gpu_h100
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --qos=gpugpu
#SBATCH -o /data/home/scyb091/run/rjy/Megatron-LM/output/slurm-%j-%x.out

set -euo pipefail
cd /data/home/scyb091/run/rjy/Megatron-LM

source /usr/share/modules/init/bash 2>/dev/null || true
source scripts/n46h2_env.sh

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
  source /usr/share/modules/init/bash 2>/dev/null || true
  source scripts/n46h2_env.sh

  export NCCL_DEBUG=INFO
  export NCCL_IB_DISABLE=0
  export NCCL_SOCKET_IFNAME=bond0
  export NCCL_IB_TIMEOUT=23
  export NCCL_IB_RETRY_CNT=7
  export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

  export NNODES='"${NNODES}"'
  export NODE_RANK="${SLURM_PROCID}"
  export NPROC_PER_NODE='"${NPROC_PER_NODE}"'
  export MASTER_ADDR='"${MASTER_ADDR}"'
  export MASTER_PORT='"${MASTER_PORT}"'
  export CUDA_VISIBLE_DEVICES=0,1,2,3

  export SCHEDULE_TEST_QWEN_MODEL_DIR=/data/home/scyb091/run/models/Qwen3-30B-A3B-Base
  export SKIP_TEST_DATA_DOWNLOAD=1

  # 5D parallel: EP=4, TP=2, CP=2, PP=2, VPP=2, ETP=1
  export SCHEDULE_TEST_EP=4
  export SCHEDULE_TEST_TP=2
  export SCHEDULE_TEST_CP=2

  echo "2n4g rank${SLURM_PROCID}: EP=4 TP=2 CP=2 PP=2 VPP=2  host=$(hostname)  WORLD=$(($NNODES * $NPROC_PER_NODE))"

  tests/unit_tests/pipeline_parallel/run_test_schedules.sh 2>&1 | tee exp_2n4g_node${SLURM_PROCID}.log
'
