import torch
from torch_geometric.nn import MessagePassing
import numpy as np
from torch_scatter import scatter_add
from numpy.random import default_rng
import torch.nn as nn

class BaseStimulation(MessagePassing):
    def __init__(self):
        super(BaseStimulation, self).__init__()
        
    def propagate(self, stimuli, targets, n_neurons):
        return scatter_add(stimuli*torch.ones_like(targets), targets, dim=0, dim_size=n_neurons)

    def stimulate(self, t):
        """Returns the stimulation at time t"""
        raise NotImplementedError

    def forward(self, t, targets, n_neurons):
        if self.duration <= t or t < 0:
            return torch.zeros((n_neurons,), device=targets.device)
        return self.propagate(stimuli=self.stimulate(t), targets=targets, n_neurons=n_neurons)