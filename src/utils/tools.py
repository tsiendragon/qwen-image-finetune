import torch


def sample_indices_per_rank(accelerator, dataset_size: int, num_samples: int,
                            *, seed: int = 0, replacement: bool = False,
                            global_shuffle: bool = True):
    """
    返回当前 rank 的 num_samples 个索引（默认无放回、不与其他 rank 重叠）。
    global_shuffle=True 时先做全局 randperm 再 stride 切片以避免排序偏置。
    """
    rank = accelerator.process_index
    world = accelerator.num_processes

    # 1) 构造该 rank 的候选池（不重叠）
    if global_shuffle:
        g0 = torch.Generator().manual_seed(seed)         # 所有 rank 相同 -> 相同 perm
        perm = torch.randperm(dataset_size, generator=g0)
        pool = perm[rank::world]
    else:
        pool = torch.arange(rank, dataset_size, world)

    # 2) 在各自池内抽样
    g = torch.Generator().manual_seed(seed + rank)       # 各 rank 不同打乱
    if replacement:
        idx = pool[torch.randint(len(pool), (num_samples,), generator=g)]
    else:
        if num_samples > len(pool):
            raise ValueError(f"rank{rank}: need {num_samples}, but only {len(pool)} available. "
                             f"Set replacement=True or reduce num_samples.")
        perm_local = torch.randperm(len(pool), generator=g)
        idx = pool[perm_local[:num_samples]]

    return idx.tolist()
