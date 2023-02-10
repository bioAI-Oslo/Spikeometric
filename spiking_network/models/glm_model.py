from spiking_network.models.base_model import BaseModel
from torch_sparse import SparseTensor
from torch_sparse import matmul
import torch
import torch.nn as nn
from tqdm import tqdm

class GLMModel(BaseModel):
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
            "threshold": 5. if "threshold" not in params else params["threshold"],
        }

        self.params = self._init_parameters(parameters, tuneable_parameters, device)

    def _init_state(self, n_neurons:int , time_scale: int) -> torch.Tensor:
        """
        Initialize the state of the neurons

        Parameters
        ----------
        n_neurons : int
            The number of neurons in the network
        time_scale : int
            The number of time steps back in time to consider

        Returns
        -------
        state : torch.Tensor [n_neurons, time_scale]
            The initial state of the network
        """
        x_initial = torch.zeros(n_neurons, time_scale, device=self.device)
        x_initial[:, time_scale-1] = torch.randint(0, 2, (n_neurons,), generator=self._rng, device=self.device)
        return x_initial
    
    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Compute the activation passed from from x_j to x_i.

        Parameters
        ----------
        x_j : torch.Tensor [n_edges, time_scale]
            The state of the neurons at the previous time_scale time steps.
        W : torch.Tensor [n_edges, time_scale]
            The weights of the edges

        Returns
        -------
        message : torch.Tensor [n_edges, 1]
            
        """
        return torch.sum(x_j * W, dim=1, keepdim=True)

    def _spike_probability(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability that a neuron spikes given its activation

        Parameters
        ----------
        activation : torch.Tensor [n_neurons, 1]
            The activation of the neurons

        Returns
        -------
        probabilities : torch.Tensor [n_neurons, 1]
            The probability that a neuron spikes
        """
        return torch.sigmoid(activation - self.params["threshold"])

    def _update_state(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Update the state of the neurons

        Parameters
        ----------
        activation : torch.Tensor [n_neurons, 1]
            The activation of the neurons

        Returns
        -------
        state : torch.Tensor [n_neurons, 1]
            The updated state of the neurons at the current time step
        """
        probabilities = self._spike_probability(activation)
        return torch.bernoulli(probabilities, generator=self._rng)
    
    def connectivity_filter(self, W0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute the connectivity filter for the spiking model given the initial weights W0 and the edge index

        Parameters
        ----------
        W0 : torch.Tensor [n_edges,]
            The initial weights of the edges
        edge_index : torch.Tensor [2, n_edges]
            The edge index

        Returns
        -------
        W : torch.Tensor [n_edges, time_scale]
            The connectivity filter
        """
        i, j = edge_index
        t = torch.arange(self.params["time_scale"], device=W0.device)
        t = t.repeat(W0.shape[0], 1) # Time steps
        is_self_edge = (i==j).unsqueeze(1).repeat(1, self.params["time_scale"]) # Is the edge a self-edge?

        # Compute the connectivity matrix
        self_edges = self._self_edges(t)
        other_edges = self._other_edges(W0, t)

        W = torch.where(is_self_edge, self_edges, other_edges).flip(1)

        return W.to(dtype=torch.float32)

    def _self_edges(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the self-edges for the spiking model.
        The first abs_ref_scale time steps are absolute refractory, the next rel_ref_scale time steps are relative refractory.
        During the absolute refractory period, the neuron is always at rest, so the weight is set to abs_ref_strength. This
        should be a high negative value to prevent the neuron from spiking.
        During the relative refractory period, the neuron is still unlikely to spike, so the weight is initilally set to
        rel_ref_strength, but decays exponentially with decay rate beta.

        Parameters
        ----------
        t : torch.Tensor [n_edges, time_scale]
            The time steps
        abs_ref_strength : float
            The inhibitory strength during the absolute refractory period
        rel_ref_strength : float
            The inhibitory strength during the relative refractory period
        abs_ref_scale : int
            The number of time steps in the absolute refractory period
        rel_ref_scale : int
            The number of time steps in the relative refractory period
        beta : float
            The decay rate of the relative refractory period

        Returns
        -------
        self_edges : torch.Tensor [n_edges, time_scale]
            The time dependent weights for the self-edges
        """
        abs_ref = self.params["abs_ref_strength"] * (t < self.params["abs_ref_scale"])
        rel_ref = (
                self.params["rel_ref_strength"] * torch.exp(-torch.abs(self.params["beta"]) * (t - self.params["abs_ref_scale"]))
                * (self.params["abs_ref_scale"] <= t) * (t <= self.params["abs_ref_scale"] + self.params["rel_ref_scale"])
            )
        return abs_ref + rel_ref

    def _other_edges(self, W0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the other-edges for the spiking model.
        The weight decays exponentially with decay rate alpha.

        Parameters
        ----------
        W0 : torch.Tensor [n_edges,]
            The initial weights of the edges
        t : torch.Tensor [n_edges, time_scale]
            The time steps
        alpha : float
            The decay rate of the edges

        Returns
        -------
        other_edges : torch.Tensor [n_edges, time_scale]
            The time dependent weights for the edges
        """

        return (
                torch.einsum("i, ij -> ij", W0, torch.exp(-torch.abs(self.params["alpha"]) * t)
                * (t < self.params["influence_scale"]))
            )
