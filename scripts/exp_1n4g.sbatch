#!/bin/bash
# Experiment: 1 node × 4 GPU (EP=2,TP=1,CP=2)
#SBATCH -J exp_1n4g
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH -o /data/home/scyb091/run/rjy/Megatron-LM/output/slurm-%j-%x.out

set -euo pipefail
cd /data/home/scyb091/run/rjy/Megatron-LM

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

export SCHEDULE_TEST_QWEN_MODEL_DIR=/data/home/scyb091/run/models/Qwen3-30B-A3B-Base
export SKIP_TEST_DATA_DOWNLOAD=1

# 5D parallel: EP=2, TP=1, CP=2, PP=2, VPP=2, ETP=1
export SCHEDULE_TEST_EP=2
export SCHEDULE_TEST_TP=1
export SCHEDULE_TEST_CP=2

echo "1n4g: EP=2 TP=1 CP=2 PP=2 VPP=2  host=$(hostname)  WORLD=$(($NNODES * $NPROC_PER_NODE))"

tests/unit_tests/pipeline_parallel/run_test_schedules.sh 2>&1 | tee exp_1n4g.log
