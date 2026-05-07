#!/bin/bash

apt install -y libibmad5
pip install pytest-mock
pip install PyGithub

cd /workspace/Megatron-LM-AutoTuner

pip install --no-deps -e .
pip install --no-deps -e verl
pip install --no-deps -e Megatron-LM-Enhanced
pip install nvidia-mathdx

export NVSHMEM_HOME=/usr/local/nvshmem
export NVTE_ENABLE_NVSHMEM=1
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/nvshmem/lib/python:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="12.0;12.0+PTX"
export NVTE_CUDA_ARCHITECTURES="120"

cd /workspace/Megatron-LM-AutoTuner/TransformerEngine-Enhanced
rm -rf build
rm -rf transformer_engine.so
rm -rf tranformer_engine_pytorch.*
rm -rf CMakeCache.txt
rm -rf CMakeFiles/
rm -rf *.egg-info/

NVTE_ENABLE_NVSHMEM=1 \
    NVSHMEM_HOME=/usr/local/nvshmem \
    NVTE_FRAMEWORK=pytorch \
    pip install --no-build-isolation -e . -vvv
cd ..

cd DeepEP-Universal
rm -rf build
rm -rf dist
rm -rf *.egg-info
source scripts/env.sh
export NVSHMEM_SYMMETRIC_SIZE=10000000000
pip uninstall deep_ep
NVSHMEM_DIR=/usr/local/nvshmem DISABLE_SM90_FEATURES=0 DISABLE_AGGRESSIVE_PTX_INSTRS=1 TORCH_CUDA_ARCH_LIST="12.0" python setup.py install
cd ..
