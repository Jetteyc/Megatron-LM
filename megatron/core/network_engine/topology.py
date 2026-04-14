import os
from typing import Iterable, Optional

import torch.distributed as dist


def get_local_world_size() -> int:
    return max(int(os.getenv("LOCAL_WORLD_SIZE", "1")), 1)


def are_ranks_in_single_node(
    ranks: Iterable[int], local_world_size: Optional[int] = None
) -> bool:
    rank_list = [int(rank) for rank in ranks]
    if not rank_list:
        return True
    lws = get_local_world_size() if local_world_size is None else max(int(local_world_size), 1)
    node_ids = {rank // lws for rank in rank_list}
    return len(node_ids) == 1


def get_group_global_ranks(group: dist.ProcessGroup) -> list[int]:
    if hasattr(dist, "get_process_group_ranks"):
        return [int(rank) for rank in dist.get_process_group_ranks(group)]

    world_rank = dist.get_rank()
    gathered_ranks = [None] * group.size()
    dist.all_gather_object(gathered_ranks, world_rank, group=group)
    return [int(rank) for rank in gathered_ranks]
