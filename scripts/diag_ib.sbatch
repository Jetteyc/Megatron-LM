#!/bin/bash
# 诊断脚本：检查计算节点的 IB / PCIe / 网络拓扑
# Usage: sbatch -p <partition> scripts/diag_ib.sbatch
# 支持 gpu_a800, gpu_h100, gpu_h200 三种分区
#SBATCH -J diag_ib
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /data/home/scyb091/run/rjy/Megatron-LM/output/diag_%j_%x.out

set -euo pipefail

source /usr/share/modules/init/bash 2>/dev/null || true

echo "=========================================="
echo "NODE: $(hostname)"
echo "PARTITION: $SLURM_JOB_PARTITION"
echo "JOB_ID: $SLURM_JOB_ID"
echo "GPU_TYPE: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "GPU_COUNT_AVAILABLE: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "=========================================="

echo ""
echo "=== 1. InfiniBand 设备 ==="
ls /sys/class/infiniband/ 2>/dev/null || echo "  -> 无 IB 设备 (/sys/class/infiniband/ 为空)"

echo ""
echo "=== 2. ibstat ==="
ibstat 2>/dev/null || echo "  -> ibstat 不可用 (无 IB)"

echo ""
echo "=== 3. ibv_devinfo ==="
ibv_devinfo 2>/dev/null || echo "  -> ibv_devinfo 不可用"

echo ""
echo "=== 4. 网络接口 ==="
ip -brief link show 2>/dev/null || ifconfig -a

echo ""
echo "=== 5. Mellanox/IB PCIe 设备 ==="
lspci 2>/dev/null | grep -iE 'mellanox|infiniband|network' || echo "  -> 未发现 Mellanox/IB PCIe 设备"

echo ""
echo "=== 6. NVLink / PCIe 拓扑 (nvidia-smi topo -m) ==="
# 有多少卡就显示多少卡的拓扑（SLURM 只分配了 1 卡，但物理机上可能有更多）
# 用 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 尝试显示完整拓扑
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nvidia-smi topo -m 2>/dev/null || echo "  -> nvidia-smi topo 不可用"

echo ""
echo "=== 7. GPU P2P 能力 ==="
nvidia-smi topo -p2p n 2>/dev/null || echo "  -> nvidia-smi topo -p2p 不可用"

echo ""
echo "=== 8. 路由表 ==="
ip route show 2>/dev/null || echo "  -> ip route 不可用"

echo ""
echo "=== 9. NCCL 环境变量 ==="
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-未设置}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-未设置}"
echo "NCCL_DEBUG=${NCCL_DEBUG:-未设置}"

echo ""
echo "=== 10. NUMA 拓扑 ==="
numactl --hardware 2>/dev/null || echo "  -> numactl 不可用"

echo ""
echo "=========================================="
echo "DIAGNOSTICS COMPLETE"
echo "=========================================="
