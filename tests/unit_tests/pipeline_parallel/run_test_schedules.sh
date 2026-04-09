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

NNODES=2
NODE_RANK=0
MASTER_ADDR=10.156.154.35
MASTER_PORT=29500

CP_INTRANODE_BACKEND=torch_dist
CP_INTERNODE_BACKEND=torch_dist
EP_INTRANODE_BACKEND=torch_dist
EP_INTERNODE_BACKEND=torch_dist
TP_INTRANODE_BACKEND=torch_dist
PP_INTERNODE_BACKEND=torch_dist
DP_INTERNODE_BACKEND=torch_dist

STAGGERED_1F1B=1
DELAY_WGRAD_COMPUTE=1
NE_NVTX_DISABLE=1
NE_SCHEDULE_NVTX_ENABLE=0
NE_SCHEDULE_NODE_RECORD_FUNCTION_ENABLE=0
NE_STAGGERED_1F1B_LOG=0
NE_STAGGERED_1F1B_LOG_MAX_CALLS=128
NE_STAGGERED_1F1B_DEBUG=0
STAGGERED_1F1B_TEST_DEBUG=0

CUDA_DEVICE_MAX_CONNECTIONS=1
NVTE_FLASH_ATTN=1
NVTE_FUSED_ATTN=0
NVTE_UNFUSED_ATTN=0
NVTE_NVTX_ENABLED=0
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

TEST_TARGET=tests/unit_tests/pipeline_parallel/test_schedules.py
TEST_FILTER=test_staggered_1f1b_profiler_with_5d_parallel
# TEST_FILTER=test_baseline_1f1b_profiler_with_5d_parallel
# TEST_FILTER=test_interleaved_1f1b_profiler_without_combined_with_5d_parallel
# TEST_FILTER="test_baseline_1f1b_profiler_with_5d_parallel or test_staggered_1f1b_profiler_with_5d_parallel or test_interleaved_1f1b_profiler_without_combined_with_5d_parallel"
PYTEST_ARGS=${PYTEST_ARGS:---tb=long -rA --full-trace}

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
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN:-1}
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-0}
export NVTE_UNFUSED_ATTN=${NVTE_UNFUSED_ATTN:-0}
export NVTE_NVTX_ENABLED=${NVTE_NVTX_ENABLED:-0}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}
export STAGGERED_1F1B=${STAGGERED_1F1B:-1}
export DELAY_WGRAD_COMPUTE=${DELAY_WGRAD_COMPUTE:-1}
export NE_NVTX_DISABLE=${NE_NVTX_DISABLE:-1}
export NE_SCHEDULE_NVTX_ENABLE=${NE_SCHEDULE_NVTX_ENABLE:-0}
export NE_SCHEDULE_NODE_RECORD_FUNCTION_ENABLE=${NE_SCHEDULE_NODE_RECORD_FUNCTION_ENABLE:-0}
export NE_STAGGERED_1F1B_LOG=${NE_STAGGERED_1F1B_LOG:-0}
export NE_STAGGERED_1F1B_LOG_MAX_CALLS=${NE_STAGGERED_1F1B_LOG_MAX_CALLS:-128}
export NE_STAGGERED_1F1B_DEBUG=${NE_STAGGERED_1F1B_DEBUG:-0}
export STAGGERED_1F1B_TEST_DEBUG=${STAGGERED_1F1B_TEST_DEBUG:-0}

export CP_INTRANODE_BACKEND=${CP_INTRANODE_BACKEND:-torch_dist}
export CP_INTERNODE_BACKEND=${CP_INTERNODE_BACKEND:-torch_dist}
export EP_INTRANODE_BACKEND=${EP_INTRANODE_BACKEND:-torch_dist}
export EP_INTERNODE_BACKEND=${EP_INTERNODE_BACKEND:-torch_dist}
export TP_INTRANODE_BACKEND=${TP_INTRANODE_BACKEND:-torch_dist}
export PP_INTERNODE_BACKEND=${PP_INTERNODE_BACKEND:-torch_dist}
export DP_INTERNODE_BACKEND=${DP_INTERNODE_BACKEND:-torch_dist}

NPROC_PER_NODE=${NPROC_PER_NODE:-${DEFAULT_NPROC_PER_NODE}}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
TRACE_ROOT=${TRACE_ROOT:-${REPO_ROOT}/outputs/1f1b_profiler}
export STAGGERED_1F1B_TRACE_DIR=${STAGGERED_1F1B_TRACE_DIR:-${TRACE_ROOT}/${RUN_ID}}
export BASELINE_1F1B_TRACE_DIR=${BASELINE_1F1B_TRACE_DIR:-${TRACE_ROOT}/${RUN_ID}}
export INTERLEAVED_1F1B_TRACE_DIR=${INTERLEAVED_1F1B_TRACE_DIR:-${TRACE_ROOT}/${RUN_ID}}

mkdir -p "${STAGGERED_1F1B_TRACE_DIR}"
mkdir -p "${BASELINE_1F1B_TRACE_DIR}"
mkdir -p "${INTERLEAVED_1F1B_TRACE_DIR}"

cd "${REPO_ROOT}"

# Clean stale .pyc to prevent NameError on nodes with old __pycache__
find "${REPO_ROOT}/megatron" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

echo "[run_test_schedules] repo_root=${REPO_ROOT}"
echo "[run_test_schedules] nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE}"
echo "[run_test_schedules] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[run_test_schedules] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[run_test_schedules] trace_dir(staggered)=${STAGGERED_1F1B_TRACE_DIR}"
echo "[run_test_schedules] trace_dir(baseline)=${BASELINE_1F1B_TRACE_DIR}"
echo "[run_test_schedules] trace_dir(interleaved)=${INTERLEAVED_1F1B_TRACE_DIR}"
echo "[run_test_schedules] test_target=${TEST_TARGET} filter=${TEST_FILTER}"
echo "[run_test_schedules] pytest_args=${PYTEST_ARGS}"
echo "[run_test_schedules] delay_wgrad_compute=${DELAY_WGRAD_COMPUTE}"
echo "[run_test_schedules] debug: pythonfaulthandler=${PYTHONFAULTHANDLER} torch_cpp_stacks=${TORCH_SHOW_CPP_STACKTRACES} torch_dist_debug=${TORCH_DISTRIBUTED_DEBUG} nccl_debug=${NCCL_DEBUG}"
echo "[run_test_schedules] network_engine_comm_ownership=enabled"
echo "[run_test_schedules] stream_policy=intranode/internode/all_bandwidth managed by megatron.core.network_engine"
echo "[run_test_schedules] backends: tp=${TP_INTRANODE_BACKEND} cp_intra=${CP_INTRANODE_BACKEND} cp_inter=${CP_INTERNODE_BACKEND} ep_intra=${EP_INTRANODE_BACKEND} ep_inter=${EP_INTERNODE_BACKEND} pp=${PP_INTERNODE_BACKEND} dp=${DP_INTERNODE_BACKEND}"

exec torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m pytest -v -s "${TEST_TARGET}" -k "${TEST_FILTER}" ${PYTEST_ARGS}