import abc
from typing import Any, Generator, List, Union

import numpy as np


class BaseReplayBuffer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def append(self, sample: Any):
        pass

    @abc.abstractmethod
    def remove(self, idx: int):
        pass


class ReplayBuffer(BaseReplayBuffer):
    def __init__(self, buffer=None, max_size: int = 10**10, episodic: bool = False):
        self.max_size = max_size
        self.episodic = episodic
        self.buffer = buffer
        if self.buffer is None:
            self.buffer = []

    def __getitem__(self, idx: Union[int, List[int]]) -> Any:
        if isinstance(idx, list):
            return [self.buffer[i] for i in idx]
        return self.buffer[idx]

    def __setitem__(self, idx: Union[int, List[int]], data: Any) -> None:
        if isinstance(idx, int):
            self.buffer[idx] = data
        else:
            for i in range(len(idx)):
                self.buffer[idx[i]] = data[i]

    def sample(self, num_samples: int, replace: bool = False) -> List[Any]:
        buffer_len = len(self.buffer)
        sampled_indices = np.random.choice(
            np.arange(buffer_len), num_samples, replace=replace
        )
        return [self.buffer[i] for i in sampled_indices]

    def append(self, sample: Any) -> None:
        if len(self.buffer) > self.max_size:
            raise ValueError("Cannot add more samples! Size is max")
        self.buffer.append(sample)

    def remove(self, idx: int) -> None:
        del self.buffer[idx]

    def get(self, batch_size: int = 1) -> Generator[Any, None, None]:
        """Returns a generator for all the samples in buffer"""

        idx = 0
        buffer_len = len(self.buffer)
        while idx < buffer_len:
            yield self.buffer[idx : idx + batch_size]
            idx += batch_size

    def reset(self) -> None:
        self.buffer = []

    def size(self) -> int:
        return len(self.buffer)
