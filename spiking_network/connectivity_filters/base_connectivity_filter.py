import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseConnectivityFilter(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, W0, edge_index):
        return W0

