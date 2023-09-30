
import numpy as np
import torch

class RollingStateArray():

    # def __init__(self, feature_size: int, buffer_size: int) -> None:
    def __init__(self, initial_array: torch.Tensor) -> None:

        self._last_insert = -1

        self._buffer_size = initial_array.shape[1]

        self._buffer = initial_array.repeat(1, 2)

        # self._buffer = np.zeros((buffer_size * 2, feature_size))



    def add(self, state):

        self._last_insert = (self._last_insert + 1) % self._buffer_size

        self._buffer[:, self._last_insert] = state

        self._buffer[:, self._buffer_size + self._last_insert ] = state

    

    @property

    def states(self):

        # print(self._last_insert, self._last_insert - self._buffer_size)

        return self._buffer[:, self._last_insert + 1 : self._last_insert + self._buffer_size + 1]

