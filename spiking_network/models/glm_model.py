from spiking_network.models.base_model import BaseModel
import torch
class GLMModel(BaseModel):
    r"""
    The GLM model uses a binomial generalized linear model to step forward in time and compute the spikes of a network of neurons.
    The model holds the recipe for constructing a connectivity filter W, which describes the weights of the edges in the network and how they vary with time,
    in addition to a threshold parameter that determines the activation needed to have a >50% chance of spiking.
    
    To compute the spikes of the network at time t, we must first compute the activation of each neuron at time t.
    Let :math:`\mathcal{N}(i)` be the set of neurons that are connected to neuron i (including itself), :math:'W_{ij}' be a vector of length :math:`\text{time_scale}`
    that contains the weights of the edges between neuron i and neuron j, and :math:`x_j` be a binary vector of length :math:`\text{time_scale}`
    that contains the spikes of neuron j at all time steps back to :math:`\text{time_scale}` time steps before the current time step.
    The activtion of neuron i at time t is then given by
    .. math::
        a_i(t) = \sum_{j \in \mathcal{N}(i)} W_{ij}\cdot x_j
    The activation is then passed through a sigmoid function to compute the probability that the neuron spikes at time t, and the spikes are
    finally computed by sampling from a Bernoulli distribution with parameter :math:`p_i(t)`.

    Certain parameters of the model are tunable, and can be tuned as normal using pytorch's framework. The tunable parameters are defined in the _tunable_parameter_keys property.
    """
    def __init__(self, parameters={}, rng=None, stimulation=None):
        """
        The default parameters are defined in the _default_parameters property, but one can
        pass a dictionary of parameters to override the default parameters. The model will throw an 
        error if the parameter dictionary contains an invalid parameter.

        Parameters
        ----------
        parameters : dict
            The parameters of the model
        rng : torch.Generator
            The random number generator to use for sampling from the Bernoulli distribution.
        stimulation : Stimulation
            The stimulation to apply to the network
        """
        super().__init__(stimulation=stimulation)
        params = self._default_parameters.copy()
        params.update(parameters)

        for key, value in params.items():
            if key not in self._valid_parameters:
                raise ValueError(f"Invalid parameter key: {key}")
            elif key in ["alpha", "beta", "threshold", "abs_ref_strength", "rel_ref_strength"]:
                self.register_parameter(key, torch.nn.Parameter(torch.tensor(value, requires_grad=False)))
            else:
                self.register_buffer(key, torch.tensor(value))
        
        self._rng = rng if rng is not None else torch.Generator()
        self.requires_grad_(False)

    
    def initialize_state(self, n_neurons:int) -> torch.Tensor:
        """
        Initialize the state of the neurons

        Parameters
        ----------
        n_neurons : int
            The number of neurons in the network
        device : str
            The device to initialize the state on

        Returns
        -------
        state : torch.Tensor [n_neurons, time_scale]
            The initial state of the network
        """
        x_initial = torch.zeros(n_neurons, self.time_scale, device=self.time_scale.device)
        x_initial[:, self.time_scale-1] = torch.randint(0, 2, (n_neurons,), generator=self._rng, device=self.time_scale.device)
        return x_initial
    
    @property
    def _default_parameters(self) -> dict:
        """Return the default parameters of the model"""
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
    def _valid_parameters(self) -> list:
        """Return the valid parameters of the model"""
        return list(self._default_parameters.keys())

    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        r"""
        For every edge (j, i) computes the message :math:'W_{ji} \cdot x_j' where :math:`W_{ji}` is vector holding the weights of the edge
        between neuron j and neuron i :math:'\text{time_scale}' time steps back in time, and :math:`x_j` is a binary vector indicating 
        whether neuron j spiked for each of the preceeding :math:'\text{time_scale}' time steps.
        ----------
        x_j : torch.Tensor [n_edges, time_scale]
            The state of the neurons at the previous time_scale time steps.
        W : torch.Tensor [n_edges, time_scale]
            The weights of the edges for the previous time_scale time steps.

        Returns
        -------
        message : torch.Tensor [n_edges, 1]
            
        """
        return torch.sum(x_j * W, dim=1, keepdim=True)

    def _probability_of_spike(self, activation: torch.Tensor) -> torch.Tensor:
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
        return torch.sigmoid(activation - self.threshold)

    def update_state(self, activation: torch.Tensor) -> torch.Tensor:
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
        probabilities = self._probability_of_spike(activation)
        return torch.bernoulli(probabilities, generator=self._rng).to(dtype=torch.uint8)
    
    def connectivity_filter(self, W0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        r"""
        The connectivity filter is a tensor of dimensions [n_edges, time_scale] that determines the time dependence of the weights of the edges.
        We receive as input a weight tensor W0 with dimensions [n_edges, 1] where n_edges is the number of edges in the network, and
        an edge index with dimensions [2, n_edges] where the first row contains the source neurons and the second row contains the target neurons.
        We therefore calculate a weight tensor W that has dimensions [n_edges, time_scale] where the entry :math:'W_{it}' at index 
        [i, t] is the weight of the ith edge at time t before the current time step.

        In order to calculate the connectivity filter, we also need some parameters:
        - abs_ref_strength : float
            The strength of the absolute refractory period
        - rel_ref_strength : float
            The strength of the relative refractory period
        - abs_ref_scale : int
            The number of time steps the absolute refractory period lasts
        - rel_ref_scale : int
            The number of time steps the relative refractory period lasts
        - influence_scale : int
            The number of time steps the influence between neurons lasts
        - time_scale : int
            The number of time steps back in time to consider

        There are two types of edges in the network: edges between neurons and edges between a neuron and itself.
        The weight of an edge between two different neurons i and j at time t before the current time step is given by 
        .. math::
            W_{ij}^t = \begin{cases}
                W0_{ij} \exp(-\beta t) & \text{if } t < \text{influence_scale} \\
                0 | \text{if } t \geq \text{influence_scale}
            \end{cases}
        
        The self-edges are used to implement the absolute and relative refractory periods. 
        The weight of a self-edge at time t before the current time step is given by
        .. math::
            W_{ii}^t = \begin{cases}
                \text{abs_ref_strength} & \text{if } t < \text{abs_ref_scale} \\
                \text{rel_ref_strength} \exp(-\alpha t) & \text{if } \text{abs_ref_scale} \leq t < \text{abs_ref_scale} + \text{rel_ref_scale} \\
                0 & \text{if } t \geq \text{abs_ref_scale} + \text{rel_ref_scale}
            \end{cases}
        
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
        i, j = edge_index # Source and target neurons

        # Time steps
        t = torch.arange(self.time_scale, device=self.time_scale.device)
        t = t.repeat(W0.shape[0], 1) # [n_edges, time_scale]

        # Boolean tensor indicating whether the edge is a self-edge
        is_self_edge = (i==j).unsqueeze(1).repeat(1, self.time_scale)

        # Compute the connectivity matrix
        self_edges = self._compute_refractory_edges(t)
        other_edges = self._compute_edges(W0, t)

        W = torch.where(is_self_edge, self_edges, other_edges).flip(1) # Flip the time axis so that the most recent time step is first

        return W.to(dtype=torch.float32)

    def _compute_refractory_edges(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the self-edges for the spiking model.
        The first abs_ref_scale time steps are absolute refractory, the next rel_ref_scale time steps are relative refractory.
        During the absolute refractory period, the neuron is always at rest, so the weight is set to abs_ref_strength. This
        should be a high negative value to prevent the neuron from spiking.
        During the relative refractory period, the neuron is still unlikely to spike, so the weight is initially set to
        rel_ref_strength, but decays exponentially with decay rate beta.

        Parameters
        ----------
        t : torch.Tensor [n_edges, time_scale]
            The time steps back in time to consider
        
        Returns
        -------
        refractory_edges : torch.Tensor [n_edges, time_scale]
            The time dependent weights for the self-edges
        """
        abs_ref = self.abs_ref_strength * (t < self.abs_ref_scale) # The first abs_ref_scale time steps are absolute refractory
        rel_ref = (
                self.rel_ref_strength * torch.exp(-torch.abs(self.beta) * (t - self.abs_ref_scale))
                * (self.abs_ref_scale <= t) * (t <= self.abs_ref_scale + self.rel_ref_scale)
            ) # The next rel_ref_scale time steps are relative refractory and decay exponentially with decay rate beta
        return abs_ref + rel_ref

    def _compute_edges(self, W0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the weights along the edges.
        The weight starts at initial weight W0 and decays exponentially with decay rate alpha until the influence scale, where it drops to 0.

        Parameters
        ----------
        W0 : torch.Tensor [n_edges,]
            The initial weights of the edges
        t : torch.Tensor [n_edges, time_scale]
            The time steps

        Returns
        -------
        other_edges : torch.Tensor [n_edges, time_scale]
            The time dependent weights for the edges
        """
        return (
                torch.einsum("i, ij -> ij", W0, torch.exp(-torch.abs(self.alpha) * t)
                * (t < self.influence_scale))
            ) # Multiplies the initial weights by the exponential decay
    
