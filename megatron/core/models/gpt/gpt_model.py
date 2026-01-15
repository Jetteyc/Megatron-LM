# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT Model selector that chooses between GPTModelNormal and GPTModelModuleQueue.

This module provides the `GPTModel` class which selects the appropriate
implementation based on configuration:
- If `post_process=True` AND `pipeline_parallel > 1` AND
    `enable_module_queue=True`: use `GPTModelModuleQueue`
- Otherwise: use `GPTModelNormal`
"""

from typing import Literal, Optional

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model_module_queue import GPTModelModuleQueue
from megatron.core.models.gpt.gpt_model_normal import GPTModelNormal
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class GPTModel(GPTModelNormal):
    """GPT Model class that dynamically selects between backends."""

    def __new__(
        cls,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'mrope', 'yarn', 'none'
        ] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        """Create appropriate GPT model instance based on configuration."""
        if cls is not GPTModel:
            return super(GPTModel, cls).__new__(cls)

        use_module_queue = False
        if post_process and getattr(config, 'enable_module_queue', False):
            pp_size = 1
            if pg_collection is not None and pg_collection.pp is not None:
                pp_size = pg_collection.pp.size()
            elif parallel_state.is_initialized():
                pp_size = parallel_state.get_pipeline_model_parallel_world_size()

            if pp_size > 1:
                use_module_queue = True

        backend_class = GPTModelModuleQueue if use_module_queue else GPTModelNormal
        return backend_class(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )


__all__ = ['GPTModel', 'GPTModelNormal', 'GPTModelModuleQueue']
