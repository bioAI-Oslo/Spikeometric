import torch
from torch_scatter import scatter_add
import numpy as np
from numpy.random import default_rng
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseStimulation(ABC, nn.Module):
    def __init__(self, targets, duration, n_neurons, device):
        super(BaseStimulation, self).__init__()
        targets = targets if isinstance(targets, list) else [targets]
        self.targets = torch.tensor(targets, device=device)
        self.duration = duration
        self.n_neurons = n_neurons
        self.device = device

    def __call__(self, t):
        """Return stimulus at time t."""
        if t >= self.duration:
            return torch.zeros(len(self.targets)).unsqueeze(0)
        pass

    def distribute(self, stimuli):
        """Distribute stimuli to targets."""
        return scatter_add(stimuli, self.targets, dim=0, dim_size=self.n_neurons)

    def to(self, device):
        """Move to device."""
        self.targets = self.targets.to(device)
        self.strengths = self.strengths.to(device)
        self.stimulation_times = self.stimulation_times.to(device)
        self.device = device
        return self
