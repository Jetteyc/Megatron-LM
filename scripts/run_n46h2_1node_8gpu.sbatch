#!/bin/bash
# 单机 8 卡：勿与 --gpus=8 同时写 -c（本集群 Lua 会拒绝）；CPU 按分区 DefCpuPerGPU 自动分配。
#SBATCH -J mg_1n8g_eq_runmd
#SBATCH -p gpu_h200
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH -o /data/home/scyb091/run/rjy/Megatron-LM/output/slurm-%j-%x.out

set -euo pipefail
cd /data/home/scyb091/run/rjy/Megatron-LM

# module 命令在 sbatch 环境不可用，需先初始化
source /usr/share/modules/init/bash 2>/dev/null || true
source scripts/n46h2_env.sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0

# 单机等价于 run.md：单进程用满 8 卡
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29531}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "single node:" "$(hostname)"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

tests/unit_tests/pipeline_parallel/run_test_schedules.sh 2>&1 | tee singlenode_8gpu.log
