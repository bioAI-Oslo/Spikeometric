from spiking_network.models.base_model import BaseModel
from torch_geometric.nn import MessagePassing
import torch
from torch import nn
from tqdm import tqdm

class LNPModel(BaseModel):
    def __init__(self, params={}, tuneable_parameters=[], seed=0, device="cpu"):
        super().__init__(device)
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

    def forward(self, x:torch.Tensor, edge_index, W, activation, **kwargs):
        """
        Computes the forward pass of the model for a single time step.

        Parameters
        ----------
        x : torch.Tensor [n_neurons, time_scale]
            The state of the network at the previous time_scale time steps.
        edge_index : torch.Tensor [2, n_edges]
            The connectivity of the network.
        W : torch.Tensor [n_edges, time_scale]
            The edge weights of the connectivity filter.
        activation : torch.Tensor [n_neurons, time_scale]
            The activation of the network at the previous time_scale time steps.

        Returns
        -------
        spikes : torch.Tensor [n_neurons, time_scale]
            The spikes of the network at the current time step.
        """
        activation = activation.unsqueeze(dim=1)
        activation += x - (activation / self.params["tau"]) * self.params["dt"]
        return self.propagate(edge_index, x=activation, W=W).squeeze()

    def message(self, x_j, W):
        """
        Compute the message from x_j to x_i
        
        Parameters
        ----------
        x_j : torch.Tensor [n_edges, 1]
            The activation of the neurons at the previous time step.
        W : torch.Tensor [n_edges, 1]
            The edge weights of the connectivity filter.

        Returns
        -------
        message : torch.Tensor [n_edges, 1]
        """
        return W * x_j

    def _update_state(self, activation):
        """
        Update the state of the neurons.
        The network will spike if the activation is above the threshold.
        Explanation of the model <-- here

        Parameters
        ----------
        activation : torch.Tensor [n_neurons, time_scale]
            The activation of the neurons at the current time step.

        Returns
        -------
        spikes : torch.Tensor [n_neurons, time_scale]
            The spikes of the network at the current time step.
        """
        noise = torch.normal(0., self.params["noise_std"], size=activation.shape, device=activation.device)
        filtered_noise = torch.normal(0., 1., size=activation.shape, device=activation.device) > self.params["noise_sparsity"]
        b_term = self.params["b"] * (1 + noise * filtered_noise)
        l = self.params["r"] * activation + b_term
        spikes = l > self.params["threshold"]
        return spikes.squeeze()

    def _spike_probability(self, activation):
        return activation
    
    def _init_state(self, n_neurons, time_scale):
        """Initializes the state of the network"""
        return torch.zeros((n_neurons, time_scale), device=self.device)
    
    def connectivity_filter(self, W0, edge_index):
        return W0.unsqueeze(1)
