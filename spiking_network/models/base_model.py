from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor
import torch
import torch.nn as nn 
import numpy as np
from torch_scatter import scatter_add
from tqdm import tqdm
from abc import ABC, abstractmethod
from pathlib import Path

class BaseModel(MessagePassing, ABC):
    def __init__(self, device="cpu"):
        super(BaseModel, self).__init__(aggr='add')
        self.device = device

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the network"""
        r"""Calculates the new state of the network

        Parameters:
        ----------
        x: torch.Tensor
            The state of the network from time t - time_scale to time t [n_neurons, time_scale]
        edge_index: torch.Tensor
            The connectivity of the network [2, n_edges]
        edge_attr: torch.Tensor
            The edge weights of the connectivity filter [n_edges, time_scale]
        t: int
            The current time step
        activation: torch.Tensor
            The activation of the network from time t - time_scale to time t [n_neurons, time_scale]

        Returns:
        -------
        new_state: torch.Tensor
            The new state of the network from time t+1 - time_scale to time t+1 [n_neurons]
        """
        return self.propagate(edge_index, x=x, **kwargs).squeeze()

    @abstractmethod
    def message(self, x_j: torch.Tensor, W: torch.Tensor):
        """Calculates the activation of the neurons

        Parameters:
        ----------
        x_j: torch.Tensor
            The state of the source neurons from time t - time_scale to time t [n_edges, time_scale]
        W: torch.Tensor
            The edge weights of the connectivity filter [n_edges, time_scale]

        Returns:
        -------
        activation: torch.Tensor
            The activation of the neurons at time t[n_edges]
        """
        return torch.sum(W*x_j, dim=1, keepdim=True)

    def save(self, path):
        """Saves the model"""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path):
        """Loads the model"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found, please tune the model first")
        model = cls()
        model.load_state_dict(torch.load(path))
        return model
    
    def _init_state(self, n_neurons, time_scale):
        """Initializes the state of the network"""
        return torch.zeros((n_neurons, time_scale), device=self.device)

    @abstractmethod
    def _spike_probability(self, activation):
        """Calculates the probability of a neuron to spike"""
        pass

    @abstractmethod
    def _update_state(self, activation):
        """Updates the state of the network"""
        pass

    def _init_parameters(self, params, tuneable, device):
        """Initializes the parameters of the model"""
        return nn.ParameterDict(
                {
                key: nn.Parameter(torch.tensor(value, device=device), requires_grad=key in tuneable)
                for key, value in params.items()
            }
        )

    def connectivity_filter(self, W0, edge_index):
        return W0.unsqueeze(1)