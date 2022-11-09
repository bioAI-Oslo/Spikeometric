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

    def simulate(self, data, n_steps, stimulation=None, verbose=True) -> torch.Tensor:
        """
        Simulates the network for n_steps time steps given the connectivity.
        It is also possible to stimulate the network by passing a stimulation function.
        Returns the state of the network at each time step.

        Parameters:
        ----------
        data: torch_geometric.data.Data
        n_steps: int
        stimulation: callable
        verbose: bool

        Returns:
        -------
        x: torch.Tensor
        """
        n_neurons = data.num_nodes
        edge_index = data.edge_index
        W0 = data.W0
        W = self.connectivity_filter(W0, edge_index)
        time_scale = W.shape[1]

        if verbose:
            pbar = tqdm(range(time_scale, n_steps + time_scale), colour="#3E5641")
        else:
            pbar = range(time_scale, n_steps + time_scale)

        x = torch.zeros(n_neurons, n_steps + time_scale, device=self.device, dtype=torch.uint8)
        activation = torch.zeros((n_neurons,), device=self.device)
        x[:, :time_scale] = self._init_state(n_neurons, time_scale)
        with torch.no_grad():
            self.eval()
            for t in pbar:
                #  print()
                #  print("\n\x1b[31mForward out:", self.forward(x[:, t-time_scale:t], edge_index, W=W, t=t, activation=activation).shape, "\x1b[0m") # ]]
                #  exit()
                activation = self.forward(x[:, t-time_scale:t], edge_index, W=W, t=t, activation=activation)
                x[:, t] = self._update_state(activation + stimulation(t) if stimulation else activation)

        return x[:, time_scale:]

    def tune(self, data, firing_rate, lr = 0.01, n_steps=1000, n_epochs=100, verbose=True):
        """
        Tunes the model parameters to match the firing rate of the network.

        Parameters:
        ----------
        data: torch_geometric.data.Data
        firing_rate: torch.Tensor
        lr: float
        n_steps: int
        n_epochs: int
        verbose: bool

        Returns:
        -------
        self: BaseModel
        """
        if verbose:
            pbar = tqdm(range(n_epochs), colour="#3E5641")
        else:
            pbar = range(n_epochs)

        edge_index = data.edge_index
        W0 = data.W0
        n_neurons = data.num_nodes
        time_scale = self.connectivity_filter(W0, edge_index).shape[1]

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        firing_rate = torch.tensor(firing_rate, device=self.device)
        for epoch in pbar:
            optimizer.zero_grad()

            # Initialize the state of the network
            x = torch.zeros(n_neurons, n_steps + time_scale, device=self.device)
            activation = torch.zeros((n_neurons, n_steps + time_scale), device=self.device)
            x[:, :time_scale] = self._init_state(n_neurons, time_scale)

            for t in range(time_scale, n_steps + time_scale):
                activation[:, t] = self.forward(x[:, t-time_scale:t], edge_index, W=self.connectivity_filter(W0, edge_index), t=t, activation=activation[:, t-time_scale:t])
                x[:, t] = self._update_state(activation[:, t])

            # Compute the loss
            avg_probability_of_spike = self._spike_probability(activation[:, time_scale:]).mean()
            loss = loss_fn(avg_probability_of_spike, firing_rate)
            if verbose:
                pbar.set_description(f"Tuning... fr={avg_probability_of_spike.item():.5f}")

            # Backpropagate
            loss.backward()
            optimizer.step()
        
        return self

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
