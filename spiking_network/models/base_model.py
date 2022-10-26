from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn 
import numpy as np
from torch_scatter import scatter_add
from tqdm import tqdm
from abc import ABC, abstractmethod

class BaseModel(MessagePassing, ABC):
    def __init__(self, connectivity_filter, device="cpu"):
        super(BaseModel, self).__init__(aggr='add')
        self.connectivity_filter = connectivity_filter.to(device) if connectivity_filter else None
        self.device = device

    def simulate(self, data, n_steps, stimulation=None, verbose=True) -> torch.Tensor:
        """Simulates the network for n_steps"""
        n_neurons = data.num_nodes
        edge_index = data.edge_index
        W0 = data.W0
        W = self.connectivity_filter(W0, edge_index) if self.connectivity_filter else W0.unsqueeze(1)
        time_scale = W.shape[1]

        if not stimulation:
            stimulation = lambda t: torch.zeros((n_neurons,), device=self.device)
        if verbose:
            pbar = tqdm(range(time_scale, n_steps + time_scale), colour="#3E5641")
        else:
            pbar = range(time_scale, n_steps + time_scale)

        x = torch.zeros(n_neurons, n_steps + time_scale, device=self.device)
        activation = torch.zeros((n_neurons, n_steps + time_scale), device=self.device)
        x[:, :time_scale] = self._init_state(n_neurons, time_scale)
        with torch.no_grad():
            self.eval()
            for t in pbar:
                activation[:, t] = self.forward(x[:, t-time_scale:t], edge_index, W, t=t, activation=activation[:, t-time_scale:t])
                x[:, t] = self._update_state(activation[:, t] + stimulation(t-time_scale))

        return x[:, time_scale:]

    def tune(self, data, firing_rate, lr = 0.01, n_steps=1000, n_epochs=100, verbose=True):
        """Tunes the parameters of the network to match the desired firing rate"""
        if verbose:
            pbar = tqdm(range(n_epochs), colour="#3E5641")
        else:
            pbar = range(n_epochs)

        edge_index = data.edge_index
        W0 = data.W0
        connectivity_filter = self.connectivity_filter if self.connectivity_filter else lambda W, edge_index: W0.unsqueeze(1)
        n_neurons = data.num_nodes
        time_scale = connectivity_filter(W0, edge_index).shape[1]

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
                activation[:, t] = self.forward(x[:, t-time_scale:t], edge_index, connectivity_filter(W0, edge_index), t=t, activation=activation[:, t-time_scale:t])
                x[:, t] = self._update_state(activation[:, t])

            # Compute the loss
            avg_probability_of_spike = self._spike_probability(activation[:, time_scale:]).mean()
            loss = loss_fn(avg_probability_of_spike, firing_rate)
            if verbose:
                pbar.set_description(f"Tuning... p={avg_probability_of_spike.item():.5f}")

            # Backpropagate
            loss.backward()
            optimizer.step()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, W: torch.Tensor, **kwargs) -> torch.Tensor:
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

        Returns:
        -------
        new_state: torch.Tensor
            The new state of the network from time t+1 - time_scale to time t+1 [n_neurons]
        """
        return self.propagate(edge_index, x=x, W=W, **kwargs).squeeze()

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
        torch.save(self.state_dict(), path)

    def load(self, path):
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found, please tune the model first")
        self.load_state_dict(torch.load(path))
    
    def _init_state(self, n_neurons, time_scale):
        return torch.zeros((n_neurons, time_scale), device=self.device)

    @abstractmethod
    def _spike_probability(self, activation):
        pass

    @abstractmethod
    def _update_state(self, activation):
        pass
