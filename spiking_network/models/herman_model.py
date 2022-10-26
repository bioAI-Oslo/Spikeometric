from spiking_network.models.base_model import BaseModel
from torch_geometric.nn import MessagePassing
import torch
from torch import nn
from tqdm import tqdm

class HermanModel(BaseModel):
    def __init__(self, connectivity_filter, tuneable_parameters=[], seed=0, device="cpu"):
        super(HermanModel, self).__init__(connectivity_filter, device)
        self._seed = seed
        self._rng = torch.Generator(device=device).manual_seed(seed)

        # Parameters
        parameters = {
            "r": 0.025,
            "threshold": 1.378e-3,
            "noise_std": 0.3,           # noise amplitude
            "b": 0.001,                 # uniform feedforward input
            "noise_sparsity": 1.0,      # noise is injected with the prob that a standard normal exceeds this
            "tau": 0.01,
            "dt": 0.0001,
        }
        
        self.params = nn.ParameterDict(
            {
                key: nn.Parameter(torch.tensor(value), requires_grad=True if key in tuneable_parameters else False)
                for key, value in parameters.items()
                }
        )


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
