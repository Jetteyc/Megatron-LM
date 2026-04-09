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
from megatron.core.network_engine import get_global_network_engine
from megatron.core.network_engine.enums import ParallelDomain, TrafficClass
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import (
    get_comm_stream,
    is_pp_first_stage,
    is_pp_last_stage,
    set_streams,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


def _debug_log(message):
    if os.environ.get('STAGGERED_1F1B_TEST_DEBUG', '1') == '0':
        return
    rank = os.environ.get('RANK', '?')
    local_rank = os.environ.get('LOCAL_RANK', '?')
    print(
        f"[staggered_1f1b_test][rank={rank}][local_rank={local_rank}] {message}",
        file=sys.stderr,
        flush=True,
    )


class _ScheduleTestGPTModel(GPTModel):
    """Keep a real `GPTModel` instance type for staggered EP overlap assertions."""

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


def _torchrun_rank():
    return int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))


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


def _make_staggered_batch(seq_length, micro_batch_size, vocab_size):
    input_ids = torch.arange(seq_length, device='cuda', dtype=torch.int64)
    _debug_log(f"_make_staggered_batch arange done")
    input_ids = input_ids.unsqueeze(0).repeat(micro_batch_size, 1) % vocab_size
    position_ids = torch.arange(seq_length, device='cuda', dtype=torch.int64)
    position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
    attention_mask = torch.ones(
        (micro_batch_size, 1, seq_length, seq_length),
        device='cuda',
        dtype=torch.bool,
    )
    _debug_log(f"_make_staggered_batch DONE")
    return {
        'input_ids': input_ids,
        'labels': input_ids.clone(),
        'position_ids': position_ids,
        'attention_mask': attention_mask,
    }


def _make_staggered_data_iterator(num_microbatches, seq_length, micro_batch_size, vocab_size):
    # Pre-materialize all batches BEFORE pipeline execution starts.
    # Lazy generation (yield) inside the pipeline loop can trigger CUDA
    # memory allocation that deadlocks when NCCL streams are active.
    batches = [
        _make_staggered_batch(seq_length, micro_batch_size, vocab_size)
        for _ in range(num_microbatches)
    ]
    return iter(batches)


def _build_staggered_gpt_model(config, vocab_size, max_sequence_length):
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
    order = schedule.convert_schedule_table_to_order(
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
    use_staggered: bool,
    overlap_moe_expert_parallel_comm: bool = True,
):
    """Shared implementation for 5D-parallel 1F1B profiler tests.

    Args:
        mocker: pytest-mock fixture.
        use_staggered: If True, sets STAGGERED_1F1B=1 so that build_schedule_plan
            returns StaggeredTransformerModelChunkSchedulePlan; otherwise uses the
            baseline TransformerModelChunkSchedulePlan.
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

    if overlap_moe_expert_parallel_comm:
        tag = "staggered" if use_staggered else "baseline"
    else:
        tag = "interleaved"

    tp_size = 4
    cp_size = 2
    ep_size = 8
    etp_size = 1
    pp_size = 2
    vpp_size = 2
    seq_length = 1024
    micro_batch_size = 2
    hidden_size = 128
    num_microbatches = 32
    vocab_size = 1024
    num_warmup_steps = 2
    num_profile_steps = 3
    total_steps = num_warmup_steps + num_profile_steps

    _debug_log(
        f"{tag} test start "
        f"world_size={world_size} tp={tp_size} cp={cp_size} ep={ep_size} pp={pp_size} vpp={vpp_size}"
    )

    os.environ['STAGGERED_1F1B'] = '1' if use_staggered else '0'
    os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'
    os.environ['NVTE_FLASH_ATTN'] = '1'
    os.environ['NVTE_FUSED_ATTN'] = '0'
    os.environ['NVTE_UNFUSED_ATTN'] = '0'
    _debug_log(
        f"mode use_staggered={use_staggered} "
        f"overlap_moe_expert_parallel_comm={overlap_moe_expert_parallel_comm}"
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
    _debug_log("random seeds set to 1234")

    set_streams()
    _debug_log(
        f"set_streams done comp_stream={torch.cuda.current_stream()} comm_stream={get_comm_stream()}"
    )

    ne = get_global_network_engine()
    pp_group = parallel_state.get_pipeline_model_parallel_group(check_initialized=False)
    _, pp_traffic = ne.resolve_for_group(
        domain=ParallelDomain.PP,
        group=pp_group,
        intranode=None,
    )
    pp_stream = ne.streams.get_stream(pp_traffic)
    _debug_log(
        f"pp stream resolved traffic={pp_traffic} pp_stream={pp_stream} global_comm_stream={get_comm_stream()}"
    )
    assert pp_stream is not None
    assert get_comm_stream() is not None
    assert pp_stream == ne.streams.get_stream(TrafficClass.INTERNODE)

    tp_backend, tp_traffic = ne.resolve(ParallelDomain.TP, intranode=True)
    cp_backend, cp_traffic = ne.resolve(ParallelDomain.CP, intranode=True)
    ep_backend, ep_traffic = ne.resolve(ParallelDomain.EP, intranode=True)
    pp_backend, pp_traffic = ne.resolve(ParallelDomain.PP, intranode=False)
    dp_backend, dp_traffic = ne.resolve(ParallelDomain.DP, intranode=False)

    assert tp_backend.name == 'torch_dist'
    assert cp_backend.name == 'torch_dist' or cp_backend.name == 'nvshmem'
    assert ep_backend.name == 'torch_dist' or ep_backend.name == 'deepep'
    assert pp_backend.name == 'torch_dist'
    assert dp_backend.name == 'torch_dist'

    assert tp_traffic == TrafficClass.INTRANODE
    assert cp_traffic == TrafficClass.INTRANODE
    # DeepEP backend uses ALL_BANDWIDTH traffic class; torch_dist uses INTRANODE
    assert ep_traffic in (TrafficClass.INTRANODE, TrafficClass.ALL_BANDWIDTH), (
        f"unexpected ep_traffic={ep_traffic} for ep_backend={ep_backend.name}"
    )
    assert pp_traffic == TrafficClass.INTERNODE
    assert dp_traffic == TrafficClass.INTERNODE
    _debug_log(
        "backend resolve done "
        f"tp=({tp_backend.name},{tp_traffic.value}) cp=({cp_backend.name},{cp_traffic.value}) "
        f"ep=({ep_backend.name},{ep_traffic.value}) pp=({pp_backend.name},{pp_traffic.value}) "
        f"dp=({dp_backend.name},{dp_traffic.value})"
    )

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
        num_layers=8,
        hidden_size=hidden_size,
        num_attention_heads=4,
        ffn_hidden_size=128,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=32,
        moe_router_topk=16,
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
    try:
        from megatron.core.transformer.moe.fused_a2a import set_deepep_num_sms
        set_deepep_num_sms(0)
    except ImportError:
        pass
    model = _build_staggered_gpt_model(
        config=config,
        vocab_size=vocab_size,
        max_sequence_length=seq_length,
    )
    _debug_log("model build complete")
    for chunk in model:
        chunk.model_type = ModelType.encoder_or_decoder

    data_iterator = [
        _make_staggered_data_iterator(
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
    for start_ev, end_ev in gpu_events:
        unit_gpu_times_ms.append(start_ev.elapsed_time(end_ev))
        
    _debug_log(f"torch profiler done ({tag})")

    trace_env = (
        'STAGGERED_1F1B_TRACE_DIR'
        if use_staggered and overlap_moe_expert_parallel_comm
        else 'BASELINE_1F1B_TRACE_DIR'
        if overlap_moe_expert_parallel_comm
        else 'INTERLEAVED_1F1B_TRACE_DIR'
    )
    trace_dir = os.environ.get(trace_env)
    if trace_dir is None and not overlap_moe_expert_parallel_comm:
        trace_dir = os.environ.get('BASELINE_1F1B_TRACE_DIR')

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

        # 1. Chrome trace JSON
        prof.export_chrome_trace(f'{rank_prefix}.json')
        _debug_log(f"chrome trace exported to {rank_prefix}.json")

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
            f.write(f"use_staggered={use_staggered}\n\n")
            f.write(
                f"overlap_moe_expert_parallel_comm={overlap_moe_expert_parallel_comm}\n"
            )
            f.write(f"delay_wgrad_compute={config.delay_wgrad_compute}\n\n")
            f.write(f"## Overall Performance (excluding Unit 0)\n")
            f.write(f"Valid Total Time = {valid_total_time_ms:.4f} ms\n")
            f.write(f"Total Tokens = {total_tokens}\n")
            f.write(f"Throughput = {throughput_tps:.2f} tokens/sec\n\n")
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
def test_staggered_1f1b_profiler_with_5d_parallel(mocker):
    _run_1f1b_profiler_with_5d_parallel(mocker, use_staggered=True)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_baseline_1f1b_profiler_with_5d_parallel(mocker):
    _run_1f1b_profiler_with_5d_parallel(mocker, use_staggered=False)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_interleaved_1f1b_profiler_without_combined_with_5d_parallel(mocker):
    _run_1f1b_profiler_with_5d_parallel(
        mocker,
        use_staggered=False,
        overlap_moe_expert_parallel_comm=False,
    )
