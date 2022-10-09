from abc import ABC, abstractmethod
from torch_geometric.nn import MessagePassing
from spiking_network.connectivity_filters.abstract_connectivity_filter import AbstractConnectivityFilter
import torch
from pathlib import Path
from scipy.sparse import coo_matrix
import numpy as np

class AbstractNetwork(ABC, MessagePassing):
    def __init__(self, connectivity_filter: AbstractConnectivityFilter, seed, device, training=False):
        super().__init__(aggr='add')
        self.connectivity_filter = connectivity_filter
        self.seed = seed
        self.rng = torch.Generator().manual_seed(self.seed)
        self.device = device

        if training:
            self.params = torch.nn.ParameterDict({key : torch.nn.Parameter(torch.Tensor([value])) for key, value in self.connectivity_filter.filter_parameters.items()})
        else:
            self.params = self.connectivity_filter.filter_parameters
            self.eval()

    @property
    def spikes(self):
        """Returns the spikes of the network"""
        return torch.nonzero(self.x, as_tuple=False).T

    @abstractmethod
    def forward(self, x, t) -> torch.Tensor:
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
        pass

    def simulate(self, n_steps: int, equilibration_steps=100) -> None:
        """Simulates the network for n_steps"""
        self._prepare(n_steps, equilibration_steps) # Prepares the network for simulation
        for t in range(n_steps): # Simulates the network for n_steps
            self.x[:, self.connectivity_filter.time_scale + t] = self.forward(self.x[:, t : self.connectivity_filter.time_scale + t], t)
        self.x = self.x[:, self.connectivity_filter.time_scale:]
        self.to_device("cpu") # Moves the network to the CPU

    def _prepare(self, n_steps: int, equilibration_steps) -> None:
        """Prepares the network for simulation by initializing the spikes, sending the tensors to device and equilibrating the network"""
        self.x = self._initialize_x(n_steps, equilibration_steps) # Sets up matrix X to store spikes
        self.to_device(self.device) # Moves the network to the right device

        for t in range(equilibration_steps): # Simulates the network for equilibration_steps
            self.x[:, self.connectivity_filter.time_scale + t] = self.forward(self.x[:, t : self.connectivity_filter.time_scale + t], t)

        self.x = self.x[:, equilibration_steps:] # Removes equilibration steps from X
    
    def _initialize_x(self, n_steps: int, equilibration_steps) -> None:
        """Initializes the matrix X to store spikes, and randomly sets the initial spikes"""
        x = torch.zeros((self.connectivity_filter.n_neurons, n_steps + equilibration_steps + self.connectivity_filter.time_scale), dtype=torch.bool)
        x[:, self.connectivity_filter.time_scale - 1] = torch.randint(0, 2, (self.connectivity_filter.n_neurons,), dtype=torch.bool, generator=self.rng)
        return x
    
    # Helper methods
    def to_device(self, device):
        """Moves the network to the GPU if available"""
        self.connectivity_filter.to_device(device)
        self.x = self.x.to(device)
        self.rng = torch.Generator(device = device).manual_seed(self.seed)


    def save(self, data_path: str) -> None:
        data_path = Path(data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        spikes = self.spikes
        x = spikes[0]
        t = spikes[1]
        data = torch.ones_like(t)
        sparse_x = coo_matrix((data, (x, t)), shape=(self.connectivity_filter.W0.shape[0], self.x.shape[1]))
        np.savez_compressed(
                data_path / Path(f"{self.seed}.npz"),
                X_sparse=sparse_x,
                W=self.connectivity_filter.W,
                edge_index=self.connectivity_filter.edge_index,
                filter_params=self.connectivity_filter.filter_parameters,
                seed=self.seed,
            )

