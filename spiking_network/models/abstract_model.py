from torch_geometric.nn import MessagePassing
import torch
import numpy as np

class AbstractModel(torch.nn.Module):
    def __init__(self, W, edge_index, n_steps, seed=0, device="cpu", equilibration_steps = 100):
        super().__init__()
        self.W = W
        self.edge_index = edge_index
        self._seed = seed
        self._rng = torch.Generator().manual_seed(self._seed)
        self._spikes = torch.empty(size=(1, 2), dtype=torch.long, device=device)
        self.device = device
        self.to_device(self.device)

        self._n_steps = n_steps
        self._equilibration_steps = equilibration_steps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simulates the network for n_steps"""
        self.to_device(self.device)
        x = self._equilibrate(x)
        for t in range(self._n_steps): # Simulates the network for n_steps
            x = self._layer(x, self.edge_index, self.W)
            self._register_spikes(x, t)
        self.to_device("cpu") # Moves the network to the CPU
        return self._spikes.T

    def _register_spikes(self, x: torch.Tensor, t: int) -> None:
        for s in torch.nonzero(x[:, -1], as_tuple=False):
            if self._spikes.shape[0] == 1:
                self._spikes = torch.tensor([s, t], device=self._spikes.device).unsqueeze(0)
            spike = torch.tensor([s, t], device=self._spikes.device)
            self._spikes = torch.cat([self._spikes, spike.unsqueeze(0)], dim=0)

    def _equilibrate(self, x: torch.Tensor) -> None:
        """Prepares the network for simulation by initializing the spikes, sending the tensors to device and equilibrating the network"""
        for t in range(self._equilibration_steps): # Simulates the network for equilibration_steps
            x = self._layer(x, self.edge_index, self.W)
        return x
    
    # Helper methods
    def to_device(self, device):
        """Moves the network to the GPU if available"""
        self.W = self.W.to(device)
        self.edge_index = self.edge_index.to(device)
        self._rng = torch.Generator(device = device).manual_seed(self._seed)
        self._spikes = self._spikes.to(device)
