"""
Utilities for data loading and processing.
"""

from torch.utils.data import IterableDataset


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
