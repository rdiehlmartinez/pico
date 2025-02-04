"""
Utilities for data loading and processing.
"""

from torch.utils.data import IterableDataset
from datasets import load_dataset


class ShardedIterableDataset(IterableDataset):
    """
    A super simple implementation of a sharded iterable dataset that enables DataParallelism
    across multiple workers. Ensures that each worker gets a unique shard of the dataset.

    NOTE: Also works fine if there is only one worker.
    """

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        iterator = iter(self.dataset)
        # NOTE: Start by skipping to this worker's shard
        for _ in range(self.rank):
            next(iterator)

        # NOTE: Yield every world_size-th item
        while True:
            try:
                yield next(iterator)
                # Skip other workers' samples
                for _ in range(self.world_size - 1):
                    next(iterator)
            except StopIteration:
                break


def load_sharded_dataset(
    dataset_name: str, initial_batch_step: int, batches_per_shard: int
) -> IterableDataset:
    """
    Load a sharded dataset for a given global step.
    """

    shard_idx = initial_batch_step // batches_per_shard
    valid_files = [
        f"train-{_shard_idx}-of-10000.parquet" for _shard_idx in range(shard_idx, 10000)
    ]
    return load_dataset(
        dataset_name, split="train", streaming=True, data_files=valid_files
    )
