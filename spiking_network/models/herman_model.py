from spiking_network.models.base_model import BaseModel
from torch_geometric.nn import MessagePassing
import torch
from torch import nn
from tqdm import tqdm

class HermanModel(BaseModel):
    def __init__(self, params={}, tuneable_parameters=[], seed=0, device="cpu"):
        super(HermanModel, self).__init__(device)
        self._seed = seed
        self._rng = torch.Generator(device=device).manual_seed(seed)

        # Parameters
        parameters = {
            "r": 0.025 if "r" not in params else params["r"],
            "b": 0.001 if "b" not in params else params["b"],
            "tau": 0.01 if "tau" not in params else params["tau"],
            "dt": 0.0001 if "dt" not in params else params["dt"],
            "noise_std": 0.3 if "noise_std" not in params else params["noise_std"],
            "noise_sparsity": 1.0 if "noise_sparsity" not in params else params["noise_sparsity"],
            "threshold": 1.378e-3 if "threshold" not in params else params["threshold"],
        }

        self.params = self._init_parameters(parameters, tuneable_parameters, device)
        

    def forward(self, x, edge_index, W, activation, **kwargs):
        activation += x - (activation / self.params["tau"]) * self.params["dt"]
        return self.propagate(edge_index, x=activation, W=W).squeeze()

    def message(self, x_j, W):
        return W * x_j

    def _update_state(self, activation):
        noise = torch.normal(0., self.params["noise_std"], size=activation.shape, device=activation.device)
        filtered_noise = torch.normal(0., 1., size=activation.shape, device=activation.device) > self.params["noise_sparsity"]
        b_term = self.params["b"] * (1 + noise * filtered_noise)
        l = self.params["r"] * activation + b_term
        spikes = l > self.params["threshold"]
        return spikes.squeeze()

    def _spike_probability(self, activation):
        return activation
