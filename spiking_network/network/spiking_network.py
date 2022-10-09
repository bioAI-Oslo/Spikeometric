import torch
from spiking_network.network.abstract_network import AbstractNetwork

class SpikingNetwork(AbstractNetwork):
    def __init__(self, connectivity_filter, seed, device):
        super().__init__(connectivity_filter, seed, device)
        self.threshold = 5

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Calculates the new state of the network

        Parameters:
        ----------
        x: torch.Tensor
            The state of the network from time t - time_scale to time t [n_neurons, time_scale]
        t: int
            The current time step

        Returns:
        -------
        x_t: torch.Tensor
            The new state of the network at time t [n_neurons]
        """
        return self.propagate(edge_index=self.connectivity_filter.edge_index, x=x, W=self.connectivity_filter.W)

    def message(self, x_j: torch.Tensor, W) -> torch.Tensor:
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
        return torch.sum(x_j*W, dim=1, keepdim=True)

    def update(self, activation: torch.Tensor) -> torch.Tensor:
        """Calculates new spikes based on the activation of the neurons

        Parameters:
        ----------
        activation: torch.Tensor
            The activation of the neurons at time t [n_neurons]

        Returns:
        -------
        x_t: torch.Tensor
            The new state of the neurons at time t [n_neurons]
        """
        probs = torch.sigmoid(activation - self.threshold)
        return torch.bernoulli(probs, generator=self.rng).squeeze()
