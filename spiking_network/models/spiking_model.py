from spiking_network.models.base_model import BaseModel
import torch
import torch.nn as nn
from tqdm import tqdm

class SpikingModel(BaseModel):
    def __init__(self, params={}, tuneable_parameters=["threshold"], seed=0, device="cpu"):
        super().__init__(device=device)
        self._seed = seed
        self._rng = torch.Generator(device=device).manual_seed(seed)
        
        parameters = {
            "alpha": 0.2 if "alpha" not in params else params["alpha"],
            "beta": 0.5 if "beta" not in params else params["beta"],
            "abs_ref_strength": -100. if "abs_ref_strength" not in params else params["abs_ref_strength"],
            "rel_ref_strength": -30. if "rel_ref_strength" not in params else params["rel_ref_strength"],
            "abs_ref_scale": 3 if "abs_ref_scale" not in params else params["abs_ref_scale"],
            "rel_ref_scale": 7 if "rel_ref_scale" not in params else params["rel_ref_scale"],
            "time_scale": 10 if "time_scale" not in params else params["time_scale"],
            "influence_scale": 5 if "influence_scale" not in params else params["influence_scale"],
            "threshold": 2.5 if "threshold" not in params else params["threshold"],
        }

        self.params = self._init_parameters(parameters, tuneable_parameters, device)

    def _init_state(self, n_neurons, time_scale):
        x_initial = torch.zeros(n_neurons, time_scale, device=self.device)
        x_initial[:, time_scale-1] = torch.randint(0, 2, (n_neurons,), generator=self._rng, device=self.device)
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
    
    def connectivity_filter(self, W0, edge_index):
        i, j = edge_index
        t = torch.arange(self.params["time_scale"], device=W0.device)
        t = t.repeat(W0.shape[0], 1) # Time steps
        is_self_edge = (i==j).unsqueeze(1).repeat(1, self.params["time_scale"]) # Is the edge a self-edge?

        # Compute the connectivity matrix
        self_edges = self._self_edges(t)
        other_edges = self._other_edges(W0, t)

        W = torch.where(is_self_edge, self_edges, other_edges).flip(1)

        return W

    def _self_edges(self, t):
        abs_ref = self.params["abs_ref_strength"] * (t < self.params["abs_ref_scale"])
        rel_ref = (
                self.params["rel_ref_strength"] * torch.exp(-torch.abs(self.params["beta"]) * (t - self.params["abs_ref_scale"]))
                * (self.params["abs_ref_scale"] <= t) * (t <= self.params["abs_ref_scale"] + self.params["rel_ref_scale"])
            )
        return abs_ref + rel_ref

    def _other_edges(self, W0, t):
        return (
                torch.einsum("i, ij -> ij", W0, torch.exp(-torch.abs(self.params["alpha"]) * t)
                * (t < self.params["influence_scale"]))
            )
