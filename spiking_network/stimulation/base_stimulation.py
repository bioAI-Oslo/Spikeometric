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
            targets = [targets]
        self.targets = self._register_attribute(targets, device)
        self.n_targets = len(self.targets)
        self.total_neurons = total_neurons
        if self.targets.max() > total_neurons - 1:
            raise ValueError("Index of target neurons must be smaller than the number of neurons.")

        self.durations = self._register_attribute(durations, device)
        if any(self.durations < 0):
            raise ValueError("All durations must be positive.")
        
        self.source_node = self.total_neurons
        self.edge_index = torch.tensor([[self.source_node] * self.n_targets, self.targets], dtype=torch.long)
        
        self.device = device

    def _register_attribute(self, attr, device):
        if isinstance(attr, int):
            attr = torch.tensor([attr]*self.n_targets, device=device, dtype=torch.long)
        if isinstance(attr, float):
            attr = torch.tensor([attr]*self.n_targets, device=device, dtype=torch.float32)
        elif isinstance(attr, list):
            dtype = torch.float32 if isinstance(attr[0], float) else torch.long
            attr = torch.tensor(attr, device=device, dtype=dtype)
        elif isinstance(attr, torch.Tensor):
            attr = attr.to(device)
        else:
            raise ValueError("Attributes must be int, a list of ints or a torch.Tensor.")
        return attr

    def propagate(self, stimuli):
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

    def forward(self, t):
        if self.durations.max() <= t or t < 0:
            return torch.zeros((self.total_neurons,), device=self.device)
        stimuli = self.stimulate(t)
        return self.propagate(stimuli=stimuli)

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        y = torch.zeros((self.total_neurons, 1), device=self.device)
        y[:inputs.shape[0]] = inputs
        return y.squeeze()