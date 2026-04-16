#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

if [[ -f "${REPO_ROOT}/.secrets/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.secrets/env.sh"
elif [[ -f "${WORKSPACE_ROOT}/.secrets/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${WORKSPACE_ROOT}/.secrets/env.sh"
fi

NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-10.156.154.35}
MASTER_PORT=${MASTER_PORT:-29500}

TEST_TARGET=tests/unit_tests/pipeline_parallel/test_schedules.py
# TEST_FILTER=test_interleaved_1f1b_profiler_without_combined_with_5d_parallel
TEST_FILTER=test_staggered_1f1b_profiler_with_5d_parallel
# TEST_FILTER=${TEST_FILTER:-test_baseline_1f1b_profiler_with_5d_parallel}
# export NE_STAGGERED_1F1B_LOG=0

PYTEST_ARGS=${PYTEST_ARGS:---tb=long -rA --full-trace}
SCHEDULE_TEST_DEBUG=${SCHEDULE_TEST_DEBUG:-0}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
    DEFAULT_NPROC_PER_NODE=${#CUDA_DEVICES[@]}
else
    DEFAULT_NPROC_PER_NODE=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
fi

if [[ "${DEFAULT_NPROC_PER_NODE}" -lt 1 ]]; then
    echo "No CUDA devices available." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
export TORCH_DISABLE_ADDR2LINE=${TORCH_DISABLE_ADDR2LINE:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export SCHEDULE_TEST_DEBUG
export SCHEDULE_TEST_BWD_DONE_LOG=${SCHEDULE_TEST_BWD_DONE_LOG:-1}
export SCHEDULE_PROFILER_RANKS=${SCHEDULE_PROFILER_RANKS:-0}
export SCHEDULE_FINE_GRAINED_OFFLOAD=0
# export SCHEDULE_FINE_GRAINED_OFFLOAD=${SCHEDULE_FINE_GRAINED_OFFLOAD:-1}
export SCHEDULE_OFFLOAD_MODULES=${SCHEDULE_OFFLOAD_MODULES:-attn_norm,qkv_linear,core_attn,attn_proj,mlp_norm,expert_fc1,moe_act}

NPROC_PER_NODE=${NPROC_PER_NODE:-${DEFAULT_NPROC_PER_NODE}}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
TRACE_ROOT=${TRACE_ROOT:-${REPO_ROOT}/outputs/1f1b_profiler}
TRACE_DIR=${TRACE_DIR:-${TRACE_ROOT}/${RUN_ID}}

mkdir -p "${TRACE_DIR}"

cd "${REPO_ROOT}"

# Clean stale .pyc to prevent NameError on nodes with old __pycache__
find "${REPO_ROOT}/megatron" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

echo "[run_test_schedules] repo_root=${REPO_ROOT}"
echo "[run_test_schedules] nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE}"
echo "[run_test_schedules] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[run_test_schedules] test_target=${TEST_TARGET} filter=${TEST_FILTER}"
echo "[run_test_schedules] trace_dir=${TRACE_DIR}"
echo "[run_test_schedules] profiler_ranks=${SCHEDULE_PROFILER_RANKS} (global ranks)"
echo "[run_test_schedules] fine_grained_offload=${SCHEDULE_FINE_GRAINED_OFFLOAD} modules=${SCHEDULE_OFFLOAD_MODULES}"

exec torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m pytest -v -s "${TEST_TARGET}" -k "${TEST_FILTER}" ${PYTEST_ARGS}