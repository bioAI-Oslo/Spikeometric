from spiking_network.models.base_model import BaseModel
import torch
import torch.nn as nn
from tqdm import tqdm

class SpikingModel(BaseModel):
    def __init__(self, connectivity_filter, tuneable_parameters=["threshold"], seed=0, device="cpu"):
        super().__init__(connectivity_filter, device)
        self._seed = seed
        self._rng = torch.Generator(device=device).manual_seed(seed)

        # Learnable parameters
        self.params = nn.ParameterDict(
            {
                "threshold": nn.Parameter(torch.tensor(0.0, device=device), requires_grad="threshold" in tuneable_parameters),
            }
        )

    def _init_state(self, n_neurons, time_scale):
        x_initial = torch.zeros(n_neurons, time_scale, device=self.device)
        x_initial[:, time_scale-1] = torch.randint(0, 2, (n_neurons,), device=self.device)
        return x_initial

    def message(self, x_j, W):
        activation = torch.sum(x_j * W, dim=1, keepdim=True)
        return activation

    def _spike_probability(self, activation):
        return torch.sigmoid(activation - self.params["threshold"])

    def _update_state(self, activation):
        """Samples the spikes of the neurons"""
        probabilities = self._spike_probability(activation)
        return torch.bernoulli(probabilities, generator=self._rng)
