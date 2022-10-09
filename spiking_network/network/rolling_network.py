import torch
from spiking_network.network.spiking_network import SpikingNetwork

class RollingNetwork(SpikingNetwork):
    def __init__(self, connectivity_filter, seed, trainable=False):
        super().__init__(connectivity_filter, seed, trainable)
        self._spikes = torch.Tensor([])

    def _rolling_x(self) -> torch.Tensor:
        self.x = torch.zeros((self.connectivity_filter.n_neurons, self.connectivity_filter.time_scale), dtype=torch.int32)
        self.x[:, self.connectivity_filter.time_scale - 1] = torch.randint(0, 2, (self.connectivity_filter.n_neurons,), dtype=torch.int32, generator=self.rng)

    def _prepare(self, n_steps: int, equilibration_steps, device) -> None:
        """Prepares the network for simulation by initializing the spikes, sending the tensors to device and equilibrating the network"""
        self._rolling_x()
        if device == "cuda":
            self.to_cuda()

        for t in range(equilibration_steps): # Simulates the network for equilibration_steps
            self._next(t)

        self._spikes = torch.Tensor([])

    def _next(self, t: int) -> None:
        """Simulates the network for one step"""
        self.x[:, -1] = self.forward(self.x)
        for s in torch.nonzero(self.x[:, -1], as_tuple=False):
            self._spikes = torch.cat([self._spikes, torch.tensor([s, t], device=self._spikes.device)])
        self.x = torch.roll(self.x, -1, dims=1) # Rolls the spikes to the left

    def simulate(self, n_steps: int, data_path="", equilibration_steps=100, device="cpu") -> None:
        """Simulates the network for n_steps"""
        super().simulate(n_steps, data_path, equilibration_steps, device)

    def to_cuda(self):
        """Moves the network to the GPU if available"""
        super().to_cuda()
        self._spikes = self._spikes.to(self.x.device)

    def to_cpu(self) -> None:
        """Moves the network to the CPU"""
        super().to_cpu()
        self._spikes = self._spikes.to(self.x.device)

    @property
    def spikes(self) -> torch.Tensor:
        return self._spikes.view(-1, 2).T
