from abc import ABC, abstractmethod
from torch_geometric.nn import MessagePassing
from spiking_network.connectivity_filters.abstract_connectivity_filter import AbstractConnectivityFilter
import torch
from pathlib import Path
from scipy.sparse import coo_matrix
import numpy as np

class AbstractNetwork(ABC, MessagePassing):
    def __init__(self, connectivity_filter: AbstractConnectivityFilter, seed, trainable=False):
        super().__init__(aggr='add')
        self.connectivity_filter = connectivity_filter
        self.seed = seed
        self.rng = torch.Generator().manual_seed(self.seed)
        if trainable:
            self.params = nn.ParameterDict(connectivity_filter.params)
            self.trainable = True
        else:
            self.trainable = False

    def forward(self, x):
        return self.propagate(self.connectivity_filter.edge_index, x=x)

    @abstractmethod
    def message(self, x_j, edge_index, edge_attr):
        pass

    def simulate(self, n_steps: int, data_path="", equilibration_steps=100, device="cpu") -> None:
        """Simulates the network for n_steps"""
        self.prepare(n_steps, equilibration_steps, device) # Prepares the network for simulation
        for t in range(n_steps): # Simulates the network for n_steps
            self._next(t)
        self.to_cpu() # Moves the network to the CPU


    def prepare(self, n_steps: int, equilibration_steps, device) -> None:
        """Prepares the network for simulation by initializing the spikes, sending the tensors to device and equilibrating the network"""
        self._initialize_x(n_steps + equilibration_steps) # Sets up matrix X to store spikes
        if device == "cuda":
            self.to_cuda()

        for t in range(equilibration_steps): # Simulates the network for equilibration_steps
            self._next(t)

        self.x = self.x[:, equilibration_steps:] # Removes equilibration steps from X
    
    def _initialize_x(self, n_steps: int) -> None:
        """Initializes the matrix X to store spikes, and randomly sets the initial spikes"""
        self.x = torch.zeros((self.connectivity_filter.n_neurons, n_steps + self.connectivity_filter.time_scale), dtype=torch.float32)
        self.x[:, self.connectivity_filter.time_scale - 1] = torch.randint(0, 2, (self.connectivity_filter.n_neurons,), dtype=torch.float32, generator=self.rng)
    
    def _next(self, t: int) -> None:
        """Calculates the next step of the network"""
        x_over_last_time_steps = self.x[:, t:t+self.connectivity_filter.time_scale] # Gets the spikes from the last time_scale steps
        self.x[:, self.connectivity_filter.time_scale + t] = self.forward(x_over_last_time_steps) # Calculates the spikes for the next step

    # Helper methods
    def to_cuda(self):
        """Moves the network to the GPU if available"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.connectivity_filter.to_device(device)
        self.x = self.x.to(device)
        self.rng = torch.Generator(device = device).manual_seed(self.seed)

    def to_cpu(self) -> None:
        """Moves the network to the CPU"""
        device = "cpu"
        self.connectivity_filter.to_device(device)
        self.x = self.x.to(device)
        self.rng = torch.Generator(device = device).manual_seed(self.seed)

