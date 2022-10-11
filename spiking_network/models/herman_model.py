from spiking_network.models.abstract_model import AbstractModel
from torch_geometric.nn import MessagePassing
import torch

class HermanModel(AbstractModel):
    def __init__(self, W, edge_index, n_steps, seed=0, device="cpu", equilibration_steps=100, r=0.025, threshold=1.378e-3, b=0.001, noise_sparsity=1.5, noise_std=0.3, tau=0.01, dt=0.0001):
        super().__init__(W, edge_index, n_steps, seed, device, equilibration_steps)
        self._layer = HermanLayer(r, threshold, b, noise_sparsity, noise_std, tau, dt, self._rng)
        
    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        """Simulates the network for n_steps"""
        activation = self._equilibrate(activation)
        x = torch.zeros((len(activation), self._n_steps), dtype=torch.bool, device=self.device)
        for t in range(self._n_steps): # Simulates the network for n_steps
            x_t, activation = self._layer(activation, self.edge_index, self.W)
            x[:, t] = x_t
        self._spikes = x.nonzero().T
        self.to_device("cpu") # Moves the network to the CPU
        return self._spikes

    def _equilibrate(self, activation):
        for _ in range(self._equilibration_steps):
            _, activation = self._layer(activation, self.edge_index, self.W)
        return activation

class HermanLayer(MessagePassing):
    def __init__(self, r, threshold, b, noise_sparsity, noise_std, tau, dt, rng):
        super(HermanLayer, self).__init__(aggr='add')
        self.r = r
        self.threshold = threshold
        self.noise_std = noise_std
        self.b = b
        self.noise_sparsity = noise_sparsity
        self.tau = tau
        self.dt = dt

        self.rng = rng


    def forward(self, state: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        return self.propagate(edge_index, x=state, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor):
        return x_j * edge_attr

    def update(self, activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.normal(0., self.noise_std, size=activation.shape, device=activation.device)
        filtered_noise = torch.normal(0., 1., size=activation.shape, device=activation.device) > self.noise_sparsity
        b_term = self.b * (1 + noise * filtered_noise)
        l = self.r * activation + b_term
        spikes = l > self.threshold
        activation = activation + spikes - (activation / self.tau) * self.dt
        return spikes.squeeze(), activation
