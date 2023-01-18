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
        return scatter_add(stimuli, targets, dim=0, dim_size=n_neurons)
    
    @property 
    def parameter_dict(self) -> dict:
        """Returns the parameters of the model"""
        tunable_parameters = {key: value.data.item() for key, value in self._tunable_params.items()}
        untunable_parameters = {k: v for k, v in self.__dict__.items() if k in self._parameter_keys and k not in tunable_parameters.keys()}
        return {**tunable_parameters, **untunable_parameters}

    def _init_parameters(self, params):
        """Initializes the parameters of the model"""
        return nn.ParameterDict(
                {
                key: nn.Parameter(value, requires_grad=False)
                for key, value in params.items()
            }
        )

    def stimulate(self, t):
        """Returns the stimulation at time t"""
        raise NotImplementedError

    def forward(self, t, targets, n_neurons):
        if self.duration <= t or t < 0:
            return torch.zeros((n_neurons,))
        stimuli = self.stimulate(t) * torch.ones((targets.shape[0],))
        return self.propagate(stimuli=stimuli, targets=targets, n_neurons=n_neurons)