import torch
from torch_geometric.nn import MessagePassing
import numpy as np
from torch_scatter import scatter_add
from numpy.random import default_rng
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseStimulation(ABC, MessagePassing):
    def __init__(self, targets, durations, total_neurons, device):
        super(BaseStimulation, self).__init__()
        if isinstance(targets, int):
            targets = torch.tensor([targets])
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        if isinstance(targets, torch.Tensor):
            if len(targets.shape) == 0:
                targets = targets.unsqueeze(0)
        self.n_targets = len(targets)
        self.targets = targets
        
        if isinstance(durations, int):
            durations = torch.tensor([durations]*self.n_targets)
        if isinstance(durations, list):
            durations = torch.tensor(durations)
        if isinstance(durations, torch.Tensor):
            if len(durations.shape) == 0:
                durations = durations.unsqueeze(0)
                durations = durations.repeat(self.n_targets)
        self.durations = durations

        if self.targets.max() > total_neurons - 1:
            raise ValueError("Index of target neurons must be smaller than the number of neurons.")
        if any(self.durations < 0):
            raise ValueError("All durations must be positive.")
        
        self.total_neurons = total_neurons
        self.device = device

    def distribute(self, stimuli):
        return scatter_add(stimuli, self.targets, dim=0, dim_size=self.total_neurons)
    
    @property 
    def parameter_dict(self) -> dict:
        """Returns the parameters of the model"""
        d = {}
        for key, value in self._params.items():
            d[key] = value.data
        d["name"] = self.__class__.__name__
        return d

    def _init_parameters(self, params):
        """Initializes the parameters of the model"""
        return nn.ParameterDict(
                {
                key: nn.Parameter(value, requires_grad=False)
                for key, value in params.items()
            }
        )