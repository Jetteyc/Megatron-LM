import inspect
import json
import os
import sys
import time
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from packaging import version

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import (
    get_comm_stream,
    is_pp_first_stage,
    is_pp_last_stage,
    set_streams,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import (
    convert_schedule_table_to_order,
    get_overlap_moe_expert_parallel_comm_order,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


_DEFAULT_QWEN_MODEL_DIR = '/data/common/models/Qwen/Qwen3-30B-A3B-Base_8layers'

_SCHEDULE_TEST_MODEL_PARAM_OVERRIDES = {
    'seq_length': 2048,
    'micro_batch_size': 1,
    'hidden_size': 768,
    'num_attention_heads': 12,
    'ffn_hidden_size': 2048,
    'moe_ffn_hidden_size': 384,
    'num_microbatches': 16,
    'vocab_size': 8192,
    'num_moe_experts': 32,
}


def _debug_log(message):
    if os.environ.get('SCHEDULE_TEST_DEBUG', '1') == '0':
        return
    rank = os.environ.get('RANK', '?')
    local_rank = os.environ.get('LOCAL_RANK', '?')
    print(
        f"[schedule_test][rank={rank}][local_rank={local_rank}] {message}",
        file=sys.stderr,
        flush=True,
    )


def _bytes_to_mib(num_bytes):
    return num_bytes / (1024.0 * 1024.0)


def _schedule_test_repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def _default_schedule_test_trace_dir():
    run_id = os.environ.get('RUN_ID', time.strftime('%Y%m%d_%H%M%S'))
    return os.path.join(_schedule_test_repo_root(), 'outputs', '1f1b_profiler', run_id)


def _normalize_schedule_test_trace_dir(trace_dir):
    if not trace_dir:
        trace_dir = _default_schedule_test_trace_dir()
    elif not os.path.isabs(trace_dir):
        trace_dir = os.path.join(_schedule_test_repo_root(), trace_dir)
    return os.path.abspath(trace_dir)


def _repo_display_path(path):
    abs_path = os.path.abspath(path)
    repo_root = _schedule_test_repo_root()
    repo_name = os.path.basename(repo_root)
    try:
        rel_path = os.path.relpath(abs_path, repo_root)
    except ValueError:
        return abs_path
    if rel_path == '.':
        return repo_name
    return os.path.join(repo_name, rel_path)


def _collect_cuda_memory_stats(phase):
    stats = torch.cuda.memory_stats()
    return {
        'phase': phase,
        'allocated_bytes': int(torch.cuda.memory_allocated()),
        'reserved_bytes': int(torch.cuda.memory_reserved()),
        'max_allocated_bytes': int(torch.cuda.max_memory_allocated()),
        'max_reserved_bytes': int(torch.cuda.max_memory_reserved()),
        'active_bytes_current': int(stats.get('active_bytes.all.current', 0)),
        'active_bytes_peak': int(stats.get('active_bytes.all.peak', 0)),
        'inactive_split_bytes_current': int(stats.get('inactive_split_bytes.all.current', 0)),
        'inactive_split_bytes_peak': int(stats.get('inactive_split_bytes.all.peak', 0)),
        'requested_bytes_current': int(stats.get('requested_bytes.all.current', 0)),
        'requested_bytes_peak': int(stats.get('requested_bytes.all.peak', 0)),
        'num_alloc_retries': int(stats.get('num_alloc_retries', 0)),
        'num_ooms': int(stats.get('num_ooms', 0)),
    }


def _try_enable_cuda_memory_history(trace_alloc_max_entries=200_000, stack_depth=32):
    recorder = getattr(torch.cuda.memory, '_record_memory_history', None)
    if recorder is None:
        _debug_log('torch.cuda.memory._record_memory_history is unavailable')
        return {'enabled': False, 'reason': 'unavailable'}

    try:
        params = set(inspect.signature(recorder).parameters.keys())
    except (TypeError, ValueError) as exc:
        _debug_log(f'failed to inspect _record_memory_history signature: {exc}')
        return {'enabled': False, 'reason': f'signature_error: {exc}'}

    kwargs = {}
    if 'context' in params:
        kwargs['context'] = 'all'
    if 'stacks' in params:
        kwargs['stacks'] = 'all'
    if 'max_entries' in params:
        kwargs['max_entries'] = trace_alloc_max_entries
    elif 'trace_alloc_max_entries' in params:
        kwargs['trace_alloc_max_entries'] = trace_alloc_max_entries
    if 'stack_depth' in params:
        kwargs['stack_depth'] = stack_depth
    if 'record_context' in params:
        kwargs['record_context'] = True
    if 'trace_alloc_record_context' in params:
        kwargs['trace_alloc_record_context'] = True

    try:
        recorder(**kwargs)
        mode = 'native'
        used_kwargs = kwargs
    except TypeError:
        legacy_kwargs = {}
        if 'enabled' in params:
            legacy_kwargs['enabled'] = True
        if 'trace_alloc_max_entries' in params:
            legacy_kwargs['trace_alloc_max_entries'] = trace_alloc_max_entries
        if 'trace_alloc_record_context' in params:
            legacy_kwargs['trace_alloc_record_context'] = True
        elif 'record_context' in params:
            legacy_kwargs['record_context'] = True

        try:
            recorder(**legacy_kwargs)
            mode = 'legacy'
            used_kwargs = legacy_kwargs
        except Exception as exc:
            _debug_log(f'failed to enable CUDA memory history: {type(exc).__name__}: {exc}')
            return {'enabled': False, 'reason': repr(exc)}
    except Exception as exc:
        _debug_log(f'failed to enable CUDA memory history: {type(exc).__name__}: {exc}')
        return {'enabled': False, 'reason': repr(exc)}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    _debug_log(f'CUDA memory history enabled mode={mode} args={used_kwargs}')
    return {'enabled': True, 'mode': mode, 'args': used_kwargs}


def _try_dump_cuda_memory_snapshot(snapshot_path):
    dumper = getattr(torch.cuda.memory, '_dump_snapshot', None)
    if dumper is None:
        _debug_log('torch.cuda.memory._dump_snapshot is unavailable')
        return False, 'unavailable'

    try:
        torch.cuda.synchronize()
        dumper(snapshot_path)
        _debug_log(f'CUDA memory snapshot exported to {snapshot_path}')
        return True, None
    except Exception as exc:
        _debug_log(f'failed to dump CUDA memory snapshot: {type(exc).__name__}: {exc}')
        return False, repr(exc)


class _ScheduleTestGPTModel(GPTModel):
    """Keep a real `GPTModel` instance type for schedule profiler assertions."""

    pass


def _populate_embedding_and_position_groups(pp_group):
    """Create *new* embedding-related process groups from *pp_group* ranks."""

    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))

    pos_embd_ranks = [pp_ranks[0]]
    embd_ranks = [pp_ranks[0]]
    if pp_ranks[-1] != pp_ranks[0]:
        embd_ranks.append(pp_ranks[-1])

    pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)
    embd_pg = dist.new_group(ranks=embd_ranks)

    return pos_embd_pg, embd_pg


def _torchrun_local_rank():
    return int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))


def _should_enable_deepep(ep_size: int) -> bool:
    """Decide DeepEP usage from EP backend env vars (mirrors main.py logic).

    When EP_INTRANODE_BACKEND=deepep, use MoEFlexTokenDispatcher with DeepEP;
    otherwise fall back to alltoall dispatcher with torch.distributed.
    """
    if ep_size <= 1:
        return False
    return os.environ.get('EP_INTRANODE_BACKEND', '').strip().lower() == 'deepep'


def _load_schedule_test_model_params():
    with open(os.path.join(_DEFAULT_QWEN_MODEL_DIR, 'config.json')) as f:
        qwen_config = json.load(f)

    model_params = {
        'model_dir': _DEFAULT_QWEN_MODEL_DIR,
        'seq_length': 2048,
        'micro_batch_size': 1,
        'hidden_size': qwen_config['hidden_size'],
        'num_layers': qwen_config['num_hidden_layers'],
        'num_attention_heads': qwen_config['num_attention_heads'],
        'num_query_groups': qwen_config['num_key_value_heads'],
        'ffn_hidden_size': qwen_config['intermediate_size'],
        'moe_ffn_hidden_size': qwen_config.get('moe_intermediate_size', 384),
        'num_microbatches': 16,
        'vocab_size': qwen_config['vocab_size'],
        'num_moe_experts': qwen_config['num_experts'],
        'moe_router_topk': qwen_config['num_experts_per_tok'],
        'rotary_base': qwen_config['rope_theta'],
        'layernorm_epsilon': qwen_config['rms_norm_eps'],
        'normalization': 'RMSNorm',
        'gated_linear_unit': True,
        'activation_func': torch.nn.functional.silu,
    }
    model_params.update(_SCHEDULE_TEST_MODEL_PARAM_OVERRIDES)
    return model_params


def _initialize_model_parallel_for_torchrun(
    *,
    tensor_model_parallel_size,
    pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size,
    context_parallel_size,
    expert_model_parallel_size,
    expert_tensor_parallel_size=None,
):
    local_rank = _torchrun_local_rank()
    _debug_log(
        "initialize_model_parallel start "
        f"tp={tensor_model_parallel_size} pp={pipeline_model_parallel_size} "
        f"vpp={virtual_pipeline_model_parallel_size} cp={context_parallel_size} "
        f"ep={expert_model_parallel_size} etp={expert_tensor_parallel_size} local_rank={local_rank}"
    )
    torch.cuda.set_device(local_rank % torch.cuda.device_count())

    if not dist.is_initialized():
        _debug_log("init_process_group begin")
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(minutes=10),
        )
        _debug_log("init_process_group done")

    if parallel_state.model_parallel_is_initialized():
        _debug_log("destroy stale model parallel state")
        parallel_state.destroy_model_parallel()

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
    )
    _debug_log("initialize_model_parallel done")


def _make_profile_batch(seq_length, micro_batch_size, vocab_size):
    input_ids = torch.arange(seq_length, device='cuda', dtype=torch.int64)
    _debug_log(f"_make_profile_batch arange done")
    input_ids = input_ids.unsqueeze(0).repeat(micro_batch_size, 1) % vocab_size
    position_ids = torch.arange(seq_length, device='cuda', dtype=torch.int64)
    position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
    attention_mask = torch.ones(
        (micro_batch_size, 1, seq_length, seq_length),
        device='cuda',
        dtype=torch.bool,
    )
    _debug_log(f"_make_profile_batch DONE")
    return {
        'input_ids': input_ids,
        'labels': input_ids.clone(),
        'position_ids': position_ids,
        'attention_mask': attention_mask,
    }


def _make_profile_data_iterator(num_microbatches, seq_length, micro_batch_size, vocab_size):
    # Pre-materialize all batches BEFORE pipeline execution starts.
    # Lazy generation (yield) inside the pipeline loop can trigger CUDA
    # memory allocation that deadlocks when NCCL streams are active.
    # All synthetic microbatches are intentionally identical in this test.
    # Reuse one prebuilt batch payload to avoid multiplying large CUDA tensors
    # (especially attention masks) by `num_microbatches`.
    template_batch = _make_profile_batch(seq_length, micro_batch_size, vocab_size)
    batches = [template_batch.copy() for _ in range(num_microbatches)]
    return iter(batches)


def _build_profile_gpt_model(config, vocab_size, max_sequence_length):
    model = []
    vp_size = config.virtual_pipeline_model_parallel_size or 1
    _debug_log(
        f"build model start vp_size={vp_size} vocab_size={vocab_size} max_seq={max_sequence_length}"
    )

    for vp_stage in range(vp_size):
        block_spec = get_gpt_decoder_block_spec(
            config=config,
            use_transformer_engine=True,
            vp_stage=vp_stage,
        )
        pre_process = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
        post_process = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
        chunk = _ScheduleTestGPTModel(
            config=config,
            transformer_layer_spec=block_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            position_embedding_type='rope',
            vp_stage=vp_stage,
            share_embeddings_and_output_weights=False,
        )
        chunk.model_type = 'unit-test'
        model.append(chunk.bfloat16().cuda())
        _debug_log(
            f"build model chunk done vp_stage={vp_stage} pre_process={pre_process} post_process={post_process}"
        )

    _debug_log(f"build model done num_chunks={len(model)}")
    return model


def test_get_forward_backward_func():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_without_interleaving
    )
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=4,
    )
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )
    Utils.destroy_model_parallel()


def test_deallocate_output_tensor():
    out = torch.tensor([[1, 2, 3], [4, 5, 6]])
    schedule.deallocate_output_tensor(out)
    assert out.nelement() == 6


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "pipeline_model_parallel_size,microbatch_group_size_per_vp_stage",
    [(1, 1), (2, 2), (2, 4), (4, 4), (4, 5), (8, 9), (8, 11)],
)
@pytest.mark.parametrize("num_microbatches", [8, 32])
@pytest.mark.parametrize("virtual_pipeline_model_parallel_size", [None, 2, 4, 8])
def test_get_pipeline_parallel_order(
    pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size,
    num_microbatches,
    microbatch_group_size_per_vp_stage,
):
    if pipeline_model_parallel_size == 1 and virtual_pipeline_model_parallel_size is not None:
        return

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
    )
    num_model_chunks = (
        virtual_pipeline_model_parallel_size
        if virtual_pipeline_model_parallel_size is not None
        else 1
    )

    _, _, num_warmup_microbatches, _ = schedule.get_pp_rank_microbatches(
        num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage, False
    )
    schedule_table = schedule.get_schedule_table(
        num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage
    )
    order = convert_schedule_table_to_order(
        num_warmup_microbatches, num_model_chunks, schedule_table
    )

    assert max(order) == num_model_chunks
    assert len(order) == num_microbatches * num_model_chunks * 2
    order_cnt = {}
    accumulated_order = 0
    for o in order:
        order_cnt[o] = order_cnt.get(o, 0) + 1
        if o < 0:
            assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
        elif -o in order_cnt:
            assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        assert accumulated_order >= 0
    assert accumulated_order == 0
    assert 0 not in order_cnt
    for k, v in order_cnt.items():
        assert -k in order_cnt and order_cnt[-k] == v

    layers_per_chunk = 2
    num_layers_per_chunk = [layers_per_chunk] * num_model_chunks
    # disable wgrad compute
    overlapped_order, chunk_id_list = get_overlap_moe_expert_parallel_comm_order(
        order, num_layers_per_chunk, False
    )
    assert max(overlapped_order) == num_model_chunks * layers_per_chunk
    assert len(overlapped_order) == len(order) * layers_per_chunk
    assert len(chunk_id_list) == len(overlapped_order)
    order_cnt = {}
    accumulated_order = 0
    for o in overlapped_order:
        order_cnt[o] = order_cnt.get(o, 0) + 1
        if o < 0:
            assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
        elif -o in order_cnt:
            assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        assert accumulated_order >= 0
    assert accumulated_order == 0

    # enable wgrad compute
    overlapped_order, chunk_id_list = get_overlap_moe_expert_parallel_comm_order(
        order, num_layers_per_chunk, True
    )
    assert max(overlapped_order) == num_model_chunks * layers_per_chunk
    assert len(overlapped_order) == len(order) * layers_per_chunk * 3 // 2
    assert len(chunk_id_list) == len(overlapped_order)
    from math import ceil

    order_cnt = {}
    accumulated_order = 0
    prev_o = 0
    for o in overlapped_order:
        if ceil(o) != o:
            assert prev_o - 0.5 == o
        else:
            order_cnt[o] = order_cnt.get(o, 0) + 1
            if o < 0:
                assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
            elif -o in order_cnt:
                assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        prev_o = o
    assert accumulated_order < 0

    Utils.destroy_model_parallel()


def test_forward_backward_func_without_pipeline_parallel(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])
        dummy_data = torch.ones(1, 4)

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return model(dummy_data), loss_func

    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=range(0, 100),
        model=[model],
        num_microbatches=4,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    for i, j in zip(losses_reduced, loss_reduced_expected):
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()


def test_forward_backward_func_with_pipeline_parallel(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_without_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
    )
    config.hidden_size = hidden_size
    model.config = config

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=None,
        model=[model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        forward_only=True,
    )

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]
    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(losses_reduced)
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_forward_backward_func_with_interleaving(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    with pytest.raises(RuntimeError):
        model.model_type = ModelType.encoder_or_decoder
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100), range(0, 100)],
            model=[model, model],
            num_microbatches=7,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=512,
            forward_only=True,
        )

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=sequence_length,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_forward_backward_func_with_uneven_interleaving(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model_a = torch.nn.Linear(4, 1)
    model_b = torch.nn.Linear(8, 1)
    model_a.vp_stage = 0
    model_b.vp_stage = 1

    def set_input_tensor(input_tensor):
        return None

    model_a.set_input_tensor = set_input_tensor
    model_b.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model_a.config = config
    model_b.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model_a.model_type = ModelType.encoder_or_decoder
    model_b.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model_a, model_b],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    with pytest.raises(RuntimeError):
        model_a.model_type = ModelType.encoder_or_decoder
        model_b.model_type = ModelType.encoder_or_decoder
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100)],
            model=[model_a, model_b],
            num_microbatches=7,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=512,
            forward_only=True,
        )

    model_a.model_type = ModelType.encoder_or_decoder
    model_b.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model_a, model_b],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=sequence_length,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_forward_backward_pipelining_without_interleaving_with_custom_pgs(mocker):
    """Test that forward_backward_pipelining_without_interleaving produces the same output
    with and without explicit process group parameters."""

    # Initialize model parallel with pipeline parallelism (no interleaving)
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    def dummy_step_func(data_iterator, model):
        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    # Create model
    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def return_none(input_tensor):
        return None

    model.set_input_tensor = return_none

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
    )
    config.hidden_size = hidden_size
    config.finalize_model_grads_func = finalize_model_grads
    model.config = config

    # Mock custom_backward to avoid actual computation
    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    # Common arguments for both calls
    common_args = {
        'forward_step_func': dummy_step_func,
        'data_iterator': None,
        'model': [model],
        'num_microbatches': micro_batch_size,
        'seq_length': sequence_length,
        'micro_batch_size': micro_batch_size,
        'forward_only': True,
    }

    # First call: without providing process group parameters (they'll be created internally)
    losses_reduced_default = schedule.forward_backward_pipelining_without_interleaving(
        **common_args
    )

    grid = HyperCommGrid([2, 1, 4, 1], ["tp", "cp", "pp", "dp"])

    pp_group = grid.create_pg("pp")
    p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    pos_embd_pg = pos_embd_pg if is_pp_first_stage(pp_group) else None
    embd_pg = embd_pg if (is_pp_last_stage(pp_group) or is_pp_first_stage(pp_group)) else None
    dp_cp_group = grid.create_pg(["dp", "cp"])

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.create_pg("tp")
    pg_collection.pp = pp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.dp_cp = dp_cp_group
    pg_collection.cp = grid.create_pg("cp")

    losses_reduced_explicit = schedule.forward_backward_pipelining_without_interleaving(
        p2p_communicator=p2p_communicator, pg_collection=pg_collection, **common_args
    )

    assert len(losses_reduced_default) == len(
        losses_reduced_explicit
    ), "Output lengths should be identical"

    for i, (default_loss, explicit_loss) in enumerate(
        zip(losses_reduced_default, losses_reduced_explicit)
    ):
        assert (
            default_loss == explicit_loss
        ), f"Loss at index {i} should be identical between default and explicit PG calls"
    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_forward_backward_pipelining_with_interleaving_with_custom_pgs(mocker):
    """Test that forward_backward_pipelining_with_interleaving produces the same output
    with and without explicit process group parameters."""

    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    grid = HyperCommGrid([1, 1, 4, 2], ["tp", "cp", "pp", "dp"])
    pp_group = grid.create_pg("pp")
    p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    pos_embd_pg = pos_embd_pg if is_pp_first_stage(pp_group) else None
    embd_pg = embd_pg if (is_pp_last_stage(pp_group) or is_pp_first_stage(pp_group)) else None

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.create_pg("tp")
    pg_collection.cp = grid.create_pg("cp")
    pg_collection.pp = pp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.dp_cp = grid.create_pg(["dp", "cp"])

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
        pg_collection=pg_collection,
        p2p_communicator=p2p_communicator,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


def test_forward_backward_no_pipelining_with_custom_pgs(mocker):
    """Validate no-pipeline schedule when explicit custom PGs are provided."""

    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank_local = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank_local, {'loss_reduced': rank_local}

        dummy_inp = torch.ones(1, 4)
        return model(dummy_inp), loss_func

    # Simple model.
    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'
    model.set_input_tensor = lambda _tensor: None  # type: ignore[assignment]

    # Minimal config.
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    grid = HyperCommGrid([2, 1, 1, 4], ["tp", "cp", "pp", "dp"])

    pp_group = grid.create_pg("pp")
    tp_group = grid.create_pg("tp")
    cp_group = grid.create_pg("cp")
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    dp_cp_group = grid.create_pg(["dp", "cp"])

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = tp_group
    pg_collection.cp = cp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.pp = pp_group
    pg_collection.dp_cp = dp_cp_group

    forward_backward_func = get_forward_backward_func()
    assert forward_backward_func == schedule.forward_backward_no_pipelining

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=range(0, 10),
        model=[model],
        num_microbatches=4,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
        pg_collection=pg_collection,
    )

    expected = {'loss_reduced': Utils.rank}
    for l in losses_reduced:
        assert l['loss_reduced'] == expected['loss_reduced']

    Utils.destroy_model_parallel()


def _run_1f1b_profiler_with_5d_parallel(
    mocker,
    *,
    overlap_moe_expert_parallel_comm: bool = True,
):
    """Shared implementation for 5D-parallel 1F1B profiler tests.

    Args:
        mocker: pytest-mock fixture.
        overlap_moe_expert_parallel_comm: Whether to use the combined forward/backward
            path for MoE EP overlap. When False, this keeps interleaved 1F1B pipeline
            scheduling enabled but uses the conventional non-combined path.
    """
    if 'RANK' not in os.environ and 'LOCAL_RANK' not in os.environ:
        pytest.skip("This test is intended to run under torchrun")

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size < 8 or world_size % 8 != 0:
        pytest.skip("Requires WORLD_SIZE to be a multiple of 8 for tp=2, cp=2, ep=2, pp=2")

    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    tag = "baseline" if overlap_moe_expert_parallel_comm else "interleaved"
    trace_env = (
        'BASELINE_1F1B_TRACE_DIR'
        if overlap_moe_expert_parallel_comm
        else 'INTERLEAVED_1F1B_TRACE_DIR'
    )
    trace_dir = _normalize_schedule_test_trace_dir(os.environ.get(trace_env))

    tp_size = 2
    cp_size = 2
    ep_size = 4
    etp_size = 1
    pp_size = 2
    vpp_size = 2
    model_params = _load_schedule_test_model_params()
    seq_length = model_params['seq_length']
    micro_batch_size = model_params['micro_batch_size']
    hidden_size = model_params['hidden_size']
    num_microbatches = model_params['num_microbatches']
    vocab_size = model_params['vocab_size']
    num_warmup_steps = 2
    num_profile_steps = 3
    total_steps = num_warmup_steps + num_profile_steps

    _debug_log(
        f"{tag} test start "
        f"world_size={world_size} tp={tp_size} cp={cp_size} ep={ep_size} pp={pp_size} vpp={vpp_size}"
    )
    if model_params['model_dir'] is not None:
        _debug_log(f"using structural params from {model_params['model_dir']}")

    os.environ.pop('STAGGERED_1F1B', None)
    os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'
    os.environ['NVTE_FLASH_ATTN'] = '1'
    os.environ['NVTE_FUSED_ATTN'] = '0'
    os.environ['NVTE_UNFUSED_ATTN'] = '0'
    _debug_log(
        f"mode overlap_moe_expert_parallel_comm={overlap_moe_expert_parallel_comm}"
    )

    _initialize_model_parallel_for_torchrun(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vpp_size,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)

    set_streams()
    _debug_log(
        f"set_streams done comp_stream={torch.cuda.current_stream()} comm_stream={get_comm_stream()}"
    )

    memory_history_info = _try_enable_cuda_memory_history()
    memory_phase_stats = [_collect_cuda_memory_stats('post_stream_setup')]

    use_deepep = _should_enable_deepep(ep_size)
    moe_dispatcher_type = "flex" if use_deepep else "alltoall"
    _debug_log(
        f"EP backend decision: EP_INTRANODE_BACKEND={os.environ.get('EP_INTRANODE_BACKEND', '<unset>')} "
        f"use_deepep={use_deepep} moe_dispatcher_type={moe_dispatcher_type}"
    )

    config = TransformerConfig(
        attention_backend=AttnBackend.flash,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vpp_size,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
        deterministic_mode=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        num_layers=model_params['num_layers'],
        hidden_size=hidden_size,
        num_attention_heads=model_params['num_attention_heads'],
        ffn_hidden_size=model_params['ffn_hidden_size'],
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=model_params['num_moe_experts'],
        moe_router_topk=model_params['moe_router_topk'],
        moe_grouped_gemm=False,
        moe_layer_freq=1,
        moe_token_dispatcher_type=moe_dispatcher_type,
        moe_enable_deepep=use_deepep,
        moe_router_dtype="fp32",
        overlap_moe_expert_parallel_comm=overlap_moe_expert_parallel_comm,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
        sequence_parallel=(tp_size > 1),
        delay_wgrad_compute=overlap_moe_expert_parallel_comm,
    )
    config.num_query_groups = model_params['num_query_groups']
    config.moe_ffn_hidden_size = model_params['moe_ffn_hidden_size']
    config.rotary_base = model_params['rotary_base']
    config.layernorm_epsilon = model_params['layernorm_epsilon']
    config.normalization = model_params['normalization']
    config.gated_linear_unit = model_params['gated_linear_unit']
    config.activation_func = model_params['activation_func']
    try:
        from megatron.core.transformer.moe.fused_a2a import set_deepep_num_sms
        set_deepep_num_sms(0)
    except ImportError:
        pass
    model = _build_profile_gpt_model(
        config=config,
        vocab_size=vocab_size,
        max_sequence_length=seq_length,
    )
    _debug_log("model build complete")
    memory_phase_stats.append(_collect_cuda_memory_stats('post_model_build'))
    for chunk in model:
        chunk.model_type = ModelType.encoder_or_decoder

    data_iterator = [
        _make_profile_data_iterator(
            num_microbatches=num_microbatches * total_steps,
            seq_length=seq_length // cp_size,  # FIX: CP distributes sequence chunks locally
            micro_batch_size=micro_batch_size,
            vocab_size=vocab_size,
        )
        for _ in range(vpp_size)
    ]
    _debug_log("data iterators prepared")

    
    def forward_step_func(data_iter, model_chunk, return_schedule_plan=False):
        _debug_log(f"forward_step_func ENTER data_iter_type={type(data_iter).__name__} model={type(model_chunk).__name__}")
        _debug_log(f"forward_step_func calling next(data_iter)...")
        batch = next(data_iter)
        _debug_log(f"forward_step_func next(data_iter) returned OK")
        _debug_log(
            f"forward_step_func batch ready return_schedule_plan={return_schedule_plan} "
            f"input_shape={tuple(batch['input_ids'].shape)}"
        )

        def loss_func(output_tensor):
            if isinstance(output_tensor, (list, tuple)):
                output_tensor = output_tensor[0]
            loss = output_tensor.float().mean()
            # NOTE: Do NOT dist.all_reduce here – loss_func is only called on
            # the last pipeline stage, so a world-wide collective would deadlock
            # because pp_rank=0 never enters this code path.
            _debug_log(f"loss_func loss={loss.item():.6f}")
            return loss, {'loss_reduced': loss.detach().clone()}

        _debug_log(f"build_schedule_plan BEGIN")
        schedule_plan = model_chunk.build_schedule_plan(**batch)
        _debug_log(f"build_schedule_plan DONE")
        num_layers = getattr(schedule_plan, '_transformer_layers', None)
        if num_layers is not None:
            num_layers = len(num_layers)
        _debug_log(
            f"schedule_plan built chunk={type(model_chunk).__name__} "
            f"plan_type={type(schedule_plan).__name__} vp_stage={getattr(schedule_plan, 'vp_stage', 'na')} "
            f"num_layers={num_layers}"
        )
        if return_schedule_plan:
            return schedule_plan, loss_func
        return model_chunk(**batch), loss_func

    unit_wall_times_ms = []
    unit_gpu_times_ms = []
    gpu_events = []

    def _record_profile_unit(label, fn, *args, **kwargs):
        _debug_log(f"enter {label} ({tag})")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(torch.cuda.current_stream())

        start_time = time.perf_counter()

        result = fn(*args, **kwargs)

        unit_wall_times_ms.append((time.perf_counter() - start_time) * 1000.0)

        end_event.record(torch.cuda.current_stream())
        gpu_events.append((start_event, end_event))

        _debug_log(f"exit {label} ({tag})")
        return result

    if overlap_moe_expert_parallel_comm:
        original_fn = schedule.combined_1f1b_schedule_for_interleaved_pipelining

        def profiled_combined_fn(*args, **kwargs):
            return _record_profile_unit(
                'combined_1f1b_schedule_for_interleaved_pipelining',
                original_fn,
                *args,
                **kwargs,
            )

        mocker.patch.object(
            schedule,
            'combined_1f1b_schedule_for_interleaved_pipelining',
            side_effect=profiled_combined_fn,
        )

    forward_backward_func = get_forward_backward_func()
    assert forward_backward_func == schedule.forward_backward_pipelining_with_interleaving
    _debug_log(f"forward_backward_func={forward_backward_func.__name__}")

    def run_profiled_forward_backward_step(*, record_unit: bool):
        call_kwargs = dict(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=seq_length,
            forward_only=False,
        )
        if record_unit and not overlap_moe_expert_parallel_comm:
            return _record_profile_unit(
                'forward_backward_pipelining_with_interleaving',
                forward_backward_func,
                **call_kwargs,
            )
        return forward_backward_func(**call_kwargs)

    _debug_log(f"starting {num_warmup_steps} warmup steps ({tag})")
    for step in range(num_warmup_steps):
        _ = run_profiled_forward_backward_step(record_unit=not overlap_moe_expert_parallel_comm)
        
    torch.cuda.synchronize()
    memory_phase_stats.append(_collect_cuda_memory_stats('post_warmup'))
    torch.cuda.reset_peak_memory_stats()
    memory_phase_stats.append(_collect_cuda_memory_stats('post_warmup_peak_reset'))
    unit_wall_times_ms.clear()
    gpu_events.clear()
    _debug_log(f"warmup complete, metrics cleared.")

    _debug_log(f"torch profiler start for {num_profile_steps} steps ({tag})")
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for step in range(num_profile_steps):
                losses_reduced = run_profiled_forward_backward_step(
                    record_unit=not overlap_moe_expert_parallel_comm
                )
    except Exception as exc:
        _debug_log(f"forward_backward_func failed: {type(exc).__name__}: {exc}")
        _debug_log(traceback.format_exc())
        raise
        
    torch.cuda.synchronize()
    memory_phase_stats.append(_collect_cuda_memory_stats('post_profile'))
    for start_ev, end_ev in gpu_events:
        unit_gpu_times_ms.append(start_ev.elapsed_time(end_ev))
        
    _debug_log(f"torch profiler done ({tag})")

    unit_count = len(unit_wall_times_ms)
    assert unit_count > 0, f"No {tag} units were executed"
    local_unit_ms = sum(unit_wall_times_ms) / unit_count
    assert local_unit_ms > 0.0

    current_rank = dist.get_rank()
    is_last_pp_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=True)

    if is_last_pp_stage:
        assert len(losses_reduced) > 0
    else:
        assert losses_reduced == []
    valid_total_time_ms = sum(unit_wall_times_ms)
    total_tokens = seq_length * micro_batch_size * num_microbatches * num_profile_steps
    throughput_tps = (total_tokens / (valid_total_time_ms / 1000.0)) if valid_total_time_ms > 0 else 0.0
    _debug_log(
        f"local_unit_count={unit_count} local_unit_ms={local_unit_ms:.6f}"
    )

    summary_line = (
        f"[{tag}_1f1b_profiler] "
        f"rank={current_rank} world_size={world_size} tp={tp_size} cp={cp_size} ep={ep_size} pp={pp_size} "
        f"vpp={vpp_size} unit_count={unit_count} \n"
        f"    => Valid Total Time (Profile Steps Only): {valid_total_time_ms:.2f} ms\n"
        f"    => Throughput: {throughput_tps:.2f} tokens/sec"
    )
    print(summary_line)
    _debug_log(f"local_unit_ms={local_unit_ms:.6f}")
    _debug_log(f"losses_reduced_len={len(losses_reduced)}")

    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        rank_prefix = os.path.join(trace_dir, f'{tag}_1f1b_rank{current_rank}')
        chrome_trace_path = f'{rank_prefix}.json'
        chrome_trace_display_path = _repo_display_path(chrome_trace_path)
        memory_phase_stats.append(_collect_cuda_memory_stats('pre_export'))
        memory_snapshot_path = f'{rank_prefix}_memory_snapshot.pickle'
        memory_snapshot_display_path = _repo_display_path(memory_snapshot_path)
        memory_snapshot_dumped, memory_snapshot_error = _try_dump_cuda_memory_snapshot(
            memory_snapshot_path
        )
        memory_stats_path = f'{rank_prefix}_memory_stats.json'
        memory_summary_path = f'{rank_prefix}_memory_summary.txt'
        memory_stats_display_path = _repo_display_path(memory_stats_path)
        memory_summary_display_path = _repo_display_path(memory_summary_path)

        with open(memory_stats_path, 'w') as f:
            json.dump(
                {
                    'rank': current_rank,
                    'tag': tag,
                    'trace_dir': _repo_display_path(trace_dir),
                    'chrome_trace_path': chrome_trace_display_path,
                    'memory_history': memory_history_info,
                    'snapshot_dumped': memory_snapshot_dumped,
                    'snapshot_path': memory_snapshot_display_path if memory_snapshot_dumped else None,
                    'snapshot_error': memory_snapshot_error,
                    'memory_stats_path': memory_stats_display_path,
                    'memory_summary_path': memory_summary_display_path,
                    'phases': memory_phase_stats,
                },
                f,
                indent=2,
            )
        _debug_log(f"memory stats exported to {memory_stats_path}")

        with open(memory_summary_path, 'w') as f:
            f.write(torch.cuda.memory_summary())
        _debug_log(f"memory summary exported to {memory_summary_path}")

        # 1. Chrome trace JSON
        prof.export_chrome_trace(chrome_trace_path)
        _debug_log(f"chrome trace exported to {chrome_trace_path}")

        # 2. Profiler key averages table (kernel-level CPU/CUDA timing)
        key_avg_table = prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=100
        )
        with open(f'{rank_prefix}_key_averages.txt', 'w') as f:
            f.write(key_avg_table)
        _debug_log(f"key averages exported to {rank_prefix}_key_averages.txt")

        # 3. Summary + per-unit wall times + losses
        with open(f'{rank_prefix}_summary.txt', 'w') as f:
            f.write(f"# {tag} 1F1B profiler summary  rank={current_rank}\n\n")
            f.write(f"## Config\n")
            f.write(f"world_size={world_size} tp={tp_size} cp={cp_size} ep={ep_size} "
                    f"pp={pp_size} vpp={vpp_size}\n")
            f.write(f"seq_length={seq_length} micro_batch_size={micro_batch_size} "
                    f"hidden_size={hidden_size} num_microbatches={num_microbatches} "
                    f"vocab_size={vocab_size}\n")
            f.write(f"model_dir={model_params['model_dir']}\n")
            f.write(
                f"num_layers={model_params['num_layers']} num_attention_heads={model_params['num_attention_heads']} "
                f"num_query_groups={model_params['num_query_groups']} ffn_hidden_size={model_params['ffn_hidden_size']} "
                f"moe_ffn_hidden_size={model_params['moe_ffn_hidden_size']}\n"
            )
            f.write(
                f"num_moe_experts={model_params['num_moe_experts']} moe_router_topk={model_params['moe_router_topk']} "
                f"rotary_base={model_params['rotary_base']} layernorm_epsilon={model_params['layernorm_epsilon']}\n"
            )
            f.write(
                f"overlap_moe_expert_parallel_comm={overlap_moe_expert_parallel_comm}\n"
            )
            f.write(f"delay_wgrad_compute={config.delay_wgrad_compute}\n\n")
            f.write(f"## Overall Performance (excluding Unit 0)\n")
            f.write(f"Valid Total Time = {valid_total_time_ms:.4f} ms\n")
            f.write(f"Total Tokens = {total_tokens}\n")
            f.write(f"Throughput = {throughput_tps:.2f} tokens/sec\n\n")
            f.write(f"## CUDA memory recording\n")
            f.write(f"trace_dir={_repo_display_path(trace_dir)}\n")
            f.write(f"chrome_trace_path={chrome_trace_display_path}\n")
            f.write(f"memory_history_enabled={memory_history_info.get('enabled', False)}\n")
            f.write(f"memory_history_mode={memory_history_info.get('mode', 'na')}\n")
            f.write(f"memory_stats_path={memory_stats_display_path}\n")
            f.write(f"memory_summary_path={memory_summary_display_path}\n")
            f.write(f"memory_snapshot_dumped={memory_snapshot_dumped}\n")
            if memory_snapshot_dumped:
                f.write(f"memory_snapshot_path={memory_snapshot_display_path}\n")
            if memory_snapshot_error:
                f.write(f"memory_snapshot_error={memory_snapshot_error}\n")
            f.write(f"\n## CUDA memory phases\n")
            for phase_stats in memory_phase_stats:
                f.write(
                    "  "
                    f"{phase_stats['phase']}: "
                    f"allocated={_bytes_to_mib(phase_stats['allocated_bytes']):.2f} MiB, "
                    f"reserved={_bytes_to_mib(phase_stats['reserved_bytes']):.2f} MiB, "
                    f"max_allocated={_bytes_to_mib(phase_stats['max_allocated_bytes']):.2f} MiB, "
                    f"max_reserved={_bytes_to_mib(phase_stats['max_reserved_bytes']):.2f} MiB, "
                    f"active_current={_bytes_to_mib(phase_stats['active_bytes_current']):.2f} MiB, "
                    f"inactive_split_current={_bytes_to_mib(phase_stats['inactive_split_bytes_current']):.2f} MiB, "
                    f"requested_current={_bytes_to_mib(phase_stats['requested_bytes_current']):.2f} MiB, "
                    f"num_alloc_retries={phase_stats['num_alloc_retries']}, "
                    f"num_ooms={phase_stats['num_ooms']}\n"
                )
            f.write("\n")
            f.write(f"## Unit wall times (ms)\n")
            f.write(f"unit_count={unit_count}\n")
            f.write(f"local_unit_ms_avg={local_unit_ms:.6f}\n")
            for i, cpu_t in enumerate(unit_wall_times_ms):
                gpu_t = unit_gpu_times_ms[i]
                stall_t = cpu_t - gpu_t
                f.write(f"  unit[{i}] = CPU Wall: {cpu_t:.4f} ms | GPU Compute: {gpu_t:.4f} ms | Stall: {stall_t:.4f} ms\n")
            f.write(f"\n## Losses\n")
            f.write(f"is_last_pp_stage={is_last_pp_stage}\n")
            f.write(f"losses_reduced_len={len(losses_reduced)}\n")
            for i, lr in enumerate(losses_reduced):
                loss_val = lr.get('loss_reduced', None) if isinstance(lr, dict) else lr
                if hasattr(loss_val, 'item'):
                    loss_val = loss_val.item()
                f.write(f"  loss[{i}] = {loss_val}\n")
        _debug_log(f"summary exported to {rank_prefix}_summary.txt")

    _debug_log("destroy_model_parallel begin")
    parallel_state.destroy_model_parallel()
    _debug_log(f"{tag} test done")


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_baseline_1f1b_profiler_with_5d_parallel(mocker):
    _run_1f1b_profiler_with_5d_parallel(mocker)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_interleaved_1f1b_profiler_without_combined_with_5d_parallel(mocker):
    _run_1f1b_profiler_with_5d_parallel(
        mocker,
        overlap_moe_expert_parallel_comm=False,
    )