from spiking_network.models.abstract_model import AbstractModel
from torch_geometric.nn import MessagePassing
import torch

class SpikingModel(AbstractModel):
    def __init__(self, W, edge_index, n_steps, seed=0, device="cpu", equilibration_steps = 100):
        super().__init__(W, edge_index, n_steps, seed, device, equilibration_steps)
        self._layer = SpikingLayer(self._rng)

class SpikingLayer(MessagePassing):
    def __init__(self, rng):
        super(SpikingLayer, self).__init__(aggr='add')
        self.threshold = 5
        self.rng = rng

    def forward(self, state: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        r"""Calculates the new state of the network

        Parameters:
        ----------
        state: torch.Tensor
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
        return self.propagate(edge_index, x=state, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor):
        """Calculates the activation of the neurons

        Parameters:
        ----------
        x_j: torch.Tensor
            The state of the source neurons from time t - time_scale to time t [n_edges, time_scale]
        edge_attr: torch.Tensor
            The edge weights of the connectivity filter [n_edges, time_scale]

        Returns:
        -------
        activation: torch.Tensor
            The activation of the neurons at time t[n_edges]
        """
        return torch.sum(x_j*edge_attr, dim=1, keepdim=True)

    def update(self, activation: torch.Tensor) -> torch.Tensor:
        """Calculates new spikes based on the activation of the neurons

        Parameters:
        ----------
        activation: torch.Tensor
            The activation of the neurons at time t + 1 [n_neurons]

        Returns:
        -------
        x_{t+1}: torch.Tensor
            The state of the neurons at time t + 1 [n_neurons]
        """
        probs = torch.sigmoid(activation - self.threshold)
        spikes = torch.bernoulli(probs, generator=self.rng)
        return spikes
