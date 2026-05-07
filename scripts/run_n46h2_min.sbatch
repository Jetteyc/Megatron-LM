#!/bin/bash
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH -J min_test
#SBATCH -o /data/home/scyb091/run/rjy/Megatron-LM/slurm-%j.out

hostname
cd /data/home/scyb091/run/rjy/Megatron-LM
source scripts/n46h2_env.sh

nvidia-smi -L
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not available in this job"
x = torch.randn(2048, 2048, device="cuda")
y = torch.randn(2048, 2048, device="cuda")
z = x @ y
print("GPU matmul ok, mean=", float(z.mean().cpu()))
PY
