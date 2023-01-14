from spiking_network.models.base_model import BaseModel
import torch
class GLMModel(BaseModel):
    def __init__(self, parameters={}, seed=None, device="cpu", stimulation=None):
        super().__init__(parameters=parameters, stimulation=stimulation, device=device)
        if seed is None:
            self._rng = torch.Generator(device=device)
        else:
            self._rng = torch.Generator(device=device).manual_seed(seed)
    
    def initialize_state(self, n_neurons:int) -> torch.Tensor:
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
        x_initial = torch.zeros(n_neurons, self.time_scale, device=self.device)
        x_initial[:, self.time_scale-1] = torch.randint(0, 2, (n_neurons,), generator=self._rng, device=self.device)
        return x_initial

    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Compute the message from x_j to x_i

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

    def probability_of_spike(self, activation: torch.Tensor) -> torch.Tensor:
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
        return torch.sigmoid(activation - self._tunable_params["threshold"])

    def update_state(self, probability: torch.Tensor) -> torch.Tensor:
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
        return torch.bernoulli(probability, generator=self._rng).to(dtype=torch.uint8)
    
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
        t = torch.arange(self.time_scale, device=W0.device)
        t = t.repeat(W0.shape[0], 1) # Time steps
        is_self_edge = (i==j).unsqueeze(1).repeat(1, self.time_scale) # Is the edge a self-edge?

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
        abs_ref = self._tunable_params["abs_ref_strength"] * (t < self.abs_ref_scale)
        rel_ref = (
                self._tunable_params["rel_ref_strength"] * torch.exp(-torch.abs(self._tunable_params["beta"]) * (t - self.abs_ref_scale))
                * (self.abs_ref_scale <= t) * (t <= self.abs_ref_scale + self.rel_ref_scale)
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
                torch.einsum("i, ij -> ij", W0, torch.exp(-torch.abs(self._tunable_params["alpha"]) * t)
                * (t < self.influence_scale))
            )
    
    @property
    def _default_parameters(self):
        return {
            "alpha": 0.2,
            "beta": 0.5,
            "threshold": 5.,
            "abs_ref_strength": -100.,
            "rel_ref_strength": -30.,
            "abs_ref_scale": 3,
            "rel_ref_scale": 7,
            "influence_scale": 5,
            "time_scale": 10,
        }

    @property
    def _tunable_parameter_keys(self):
        return ["alpha", "beta", "threshold", "abs_ref_strength", "rel_ref_strength"]
