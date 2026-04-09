# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List, Optional

import torch.distributed as dist


@lru_cache(maxsize=1)
def get_local_world_size() -> int:
    return max(int(os.getenv("LOCAL_WORLD_SIZE", "1")), 1)


def is_intranode_rank_pair(rank_a: int, rank_b: int, local_world_size: Optional[int] = None) -> bool:
    lws = local_world_size if local_world_size is not None else get_local_world_size()
    return (int(rank_a) // lws) == (int(rank_b) // lws)


def are_ranks_in_single_node(ranks: Iterable[int], local_world_size: Optional[int] = None) -> bool:
    rank_list = [int(r) for r in ranks]
    if not rank_list:
        return True
    lws = local_world_size if local_world_size is not None else get_local_world_size()
    node_ids = {r // lws for r in rank_list}
    return len(node_ids) == 1


def get_group_global_ranks(group: dist.ProcessGroup) -> List[int]:
    if hasattr(dist, "get_process_group_ranks"):
        try:
            return [int(r) for r in dist.get_process_group_ranks(group)]
        except Exception:
            pass

    world_rank = dist.get_rank()
    world_ranks = [None] * group.size()
    dist.all_gather_object(world_ranks, world_rank, group=group)
    return [int(r) for r in world_ranks]


def is_group_intranode(group: dist.ProcessGroup, local_world_size: Optional[int] = None) -> bool:
    return are_ranks_in_single_node(get_group_global_ranks(group), local_world_size=local_world_size)
