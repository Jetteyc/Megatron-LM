#!/bin/bash
# 单机 4 卡：勿与 --gpus 同时写 -c（本集群 Lua 会拒绝）；CPU 按分区 DefCpuPerGPU 自动分配。
#SBATCH -J mg_1n4g_eq_runmd
#SBATCH -p gpu_a800
#SBATCH -N 1
#SBATCH --gpus=4
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

export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29531}
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "single node 4gpu:" "$(hostname)"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

tests/unit_tests/pipeline_parallel/run_test_schedules.sh 2>&1 | tee singlenode_4gpu.log
