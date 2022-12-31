from spiking_network.models.base_model import BaseModel
from torch_geometric.nn import MessagePassing
import torch

class HermanModel(BaseModel):
    def __init__(self, parameters={}, seed=0, device="cpu"):
        super(HermanModel, self).__init__(parameters, device)
        self._seed = seed
        self._rng = torch.Generator(device=device).manual_seed(seed)

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
        activation += x - (activation / self._params["tau"]) * self._params["dt"]
        activation = self.propagate(edge_index, x=activation, W=W).squeeze()
        return self.spike(activation)


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

    def spike(self, activation):
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
        noise = torch.normal(0., self._params["noise_std"], size=activation.shape, device=activation.device)
        filtered_noise = torch.normal(0., 1., size=activation.shape, device=activation.device) > self._params["noise_sparsity"]
        b_term = self._params["b"] * (1 + noise * filtered_noise)
        l = self._params["r"] * activation + b_term
        spikes = l > self._params["threshold"]
        return spikes.squeeze()

    @property
    def _default_parameters(self):
        return {
            "r": 0.025,
            "b": 0.001,
            "tau": 0.01,
            "dt": 0.0001,
            "noise_std": 0.3,
            "noise_sparsity": 1.0,
            "threshold": 1.378e-3,
            "time_scale": 1,
        }

    def probability_of_spike(self, activation):
        return activation
