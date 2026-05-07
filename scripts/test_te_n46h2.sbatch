#!/bin/bash
#SBATCH -J te_nvshmem_test
#SBATCH -p gpu_a800
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /data/home/scyb091/run/rjy/Megatron-LM/output/slurm-%j-te_test.out

set -euo pipefail
cd /data/home/scyb091/run/rjy/Megatron-LM

source /usr/share/modules/init/bash 2>/dev/null || true
source scripts/n46h2_env.sh

echo "=== hostname ==="
hostname

echo ""
echo "=== nvcc ==="
which nvcc

echo ""
echo "=== LD_LIBRARY_PATH (first 5) ==="
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | head -5

echo ""
echo "=== TE import ==="
python -c "
import transformer_engine
print('TE version:', transformer_engine.__version__)
import transformer_engine.pytorch
print('pytorch module ok')
from transformer_engine.pytorch import fp8_autocast
print('fp8_autocast ok')
import os
print('NVSHMEM_HOME:', os.environ.get('NVSHMEM_HOME', 'not set'))
print('NVTE_ENABLE_NVSHMEM:', os.environ.get('NVTE_ENABLE_NVSHMEM', 'not set'))
"

echo ""
echo "=== torch cuda test ==="
python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device count:', torch.cuda.device_count())
    print('cuda device name:', torch.cuda.get_device_name(0))
"

echo ""
echo "=== TE + torch integration ==="
python -c "
import torch
import transformer_engine.pytorch as te
x = torch.randn(2, 4, device='cuda')
ln = te.LayerNorm(4).to('cuda')
y = ln(x)
print('LayerNorm forward OK, output shape:', y.shape)
"

echo ""
echo "DONE"
