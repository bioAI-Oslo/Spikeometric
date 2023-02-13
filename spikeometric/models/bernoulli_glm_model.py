from spikeometric.models.base_model import BaseModel
import torch
import torch.nn as nn
from torch_geometric.utils import add_remaining_self_loops

class BernoulliGLM(BaseModel):
    r"""
    The Bernoulli GLM from `"Inferring causal connectivity from pairwise recordings and optogenetics" <https://www.biorxiv.org/content/10.1101/463760v3.full>`_.

    This is a Generalized Linear Model with a logit link function and a Bernoulli distributed response. 
    Intuitively, it passes the input to each neuron through a sigmoid nonlinearity to get a probability of firing and
    samples spikes from the resulting Bernoulli distribution.

    More precisely, at time :math:`t+1`, the state :math:`x_i(t+1)` of neuron :math:`i` is calculated in three steps:
    
    We start by computing the input to :math:`i`:

    .. math::
        g_i(t+1) = \sum_{j \in \mathcal{N}(i) \cup \{ i \}} \mathbf{W_{j, i}} \cdot \mathbf{x_j}(t) + f_i(t+1)

    Where the first sum represents the input from the other neurons in the network and the second term represents the input from an external stimulus.

    The product :math:`\mathbf{W_{j,i}} \cdot \mathbf{x_j}(t)` is calculated in one of two ways according to
    whether the edge is between a neuron and itself or between two different neurons.
    
    For the case :math:`j \neq i` we convolve the state of the network during the coupling window :math:`c_w` with the synaptic weights 
    :math:`(W_0)_{j,i}` using an exponential filter:

    .. math::
        \mathbf{W_{j,i}} \cdot \mathbf{x_j}(t) = \sum_{t'=0}^{c_w-1} (W_0)_{j, i} \: x_j(t-t') \: e^{- \alpha \Delta t \: t'}
    
    And for :math:`j=i` we want to add a negative input to the neuron during the absolute refractory period :math:`A_{ref}` 
    and a decaying negative input during the relative refractory period :math:`R_{ref}`:

    .. math::

        \mathbf{W_{i,i}} \cdot \mathbf{x_i}(t) = \sum_{t'=0}^{A_{ref}-1} a \: x_i(t-t') + \sum_{t'=A_{ref}}^{R_{ref}-1} r \: x_i(t-t') \: e^{- \beta \Delta t \: t'}
    
    where :math:`a` is the absolute refractory strength and :math:`r` is the relative refractory strength.

    We then convert the input to a firing rate by subtracting a threshold value :math:`\theta` and applying a sigmoid non-linearity:

    .. math::
        p_i(t+1) = \frac{1}{1 + e^{-(g_i(t+1) - \theta)}}

    Finally, we draw spikes from a Bernoulli distribution with the firing rate :math:`p_i(t)` as the probability of spiking for each neuron.

    .. math::
        x_i(t+1) \sim Bernoulli(p_i(t+1))
    
    Parameters
    -----------
    theta : float
        The threshold activation :math:`\theta` above which the neurons spike with probability > 0.5. (tunable)
    dt : float
        The time step size :math:`\Delta t` in milliseconds.
    coupling_window : int
        Length of the coupling window :math:`c_w` in time steps
    alpha : float
        The decay rate :math:`\alpha` of the negative activation during the relative refractory period (tunable)
    abs_ref_scale : int
        The absolute refractory period of the neurons :math:`A_{ref}` in time steps
    abs_ref_strength : float
        The large negative activation :math:`a` added to the neurons during the absolute refractory period
    rel_ref_scale : int
        The relative refractory period of the neurons :math:`R_{ref}` in time steps
    rel_ref_strength : float
        The negative activation :math:`r` added to the neurons during the relative refractory period (tunable)
    beta : float
        The decay rate :math:`\beta` of the weights. (tunable)
    rng : torch.Generator
        The random number generator for sampling from the Bernoulli distribution.
    """

    def __init__(self, 
            theta: float,
            dt: float,
            coupling_window: int,
            alpha: float,
            abs_ref_scale: int,
            abs_ref_strength: float,
            rel_ref_scale: int,
            rel_ref_strength: int,
            beta: float,
            rng=None
        ):
        super().__init__()
        # Buffers are used to store tensors that will not be tunable
        T = max(coupling_window, abs_ref_scale + rel_ref_scale) # The total number of time steps we need to store for the convolution

        self.register_buffer("T", torch.tensor(T, dtype=torch.int))
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))
        self.register_buffer("coupling_window", torch.tensor(coupling_window, dtype=torch.int))
        self.register_buffer("abs_ref_scale", torch.tensor(abs_ref_scale, dtype=torch.int))
        self.register_buffer("rel_ref_scale", torch.tensor(rel_ref_scale, dtype=torch.int))
        self.register_buffer("abs_ref_strength", torch.tensor(abs_ref_strength, dtype=torch.float))

        # Parameters are used to store tensors that will be tunable
        self.register_parameter("theta", nn.Parameter(torch.tensor(theta, dtype=torch.float)))
        self.register_parameter("beta", nn.Parameter(torch.tensor(beta, dtype=torch.float)))
        self.register_parameter("alpha", nn.Parameter(torch.tensor(alpha, dtype=torch.float)))
        self.register_parameter("rel_ref_strength", nn.Parameter(torch.tensor(rel_ref_strength, dtype=torch.float)))

        self._rng = rng if rng is not None else torch.Generator()
        self.requires_grad_(False)
    
    def input(self, edge_index: torch.Tensor, W: torch.Tensor, state: torch.Tensor, t=-1, stimulus_mask: torch.Tensor = 0) -> torch.Tensor:
        r"""
        Computes the input at time step :obj:`t+1` by adding together the synaptic input from neighboring neurons and the stimulus input.

        .. math::
            g_i(t+1) = \sum_{j \in \mathcal{N}(i) \cup \{ i \}} \mathbf{W_{j, i}} \cdot \mathbf{x_j}(t) + f_i(t+1)

        Parameters
        ----------
        edge_index : torch.Tensor [2, n_edges]
            The connectivity of the network
        W : torch.Tensor [n_edges, T]
            The weights of the edges
        state : torch.Tensor [n_neurons, T]
            The state of the neurons
        t : int
            The current time step
        stimulus_mask : torch.Tensor [n_stimulated_neurons, 1]
            The indices of the neurons that are stimulated

        Returns
        -------
        synaptic_input : torch.Tensor [n_neurons, 1]
        
        """
        return self.synaptic_input(edge_index, W, state=state) + self.stimulus_input(t, stimulus_mask)

    def non_linearity(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the probability that a neuron spikes given its input

        .. math::
            p_i(t) = \frac{1}{1 + e^{-(g_i(t) - \theta)}}

        Parameters
        ----------
        input : torch.Tensor [n_neurons, 1]
            The synaptic input to the neurons

        Returns
        --------
        probabilities : torch.Tensor [n_neurons, 1]
            The probability that a neuron spikes
        """
        return torch.sigmoid(input - self.theta)*self.dt

    def emit_spikes(self, probabilities: torch.Tensor) -> torch.Tensor:
        r"""
        Emits spikes from the neurons given their probabilities of spiking

        .. math::
            x_i(t) \sim Bernoulli(p_i(t))

        Parameters
        ----------
        probabilites : torch.Tensor [n_neurons, 1]
            The probability that a neuron spikes

        Returns
        -------
        spikes : torch.Tensor [n_neurons, 1]
            The spikes emitted by the neurons (1 if the neuron spikes, 0 otherwise)
        """
        return torch.bernoulli(probabilities, generator=self._rng).to(torch.int)

    def connectivity_filter(self, W0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        r"""
        The connectivity filter constructs a tensor holding the weights of the edges in the network.
        These weights represent the strength of the connection between two neurons over the coupling window 
        and are used to compute the synaptic input.

        There are two types of edges: edges between different neurons and edges between a neuron and itself.

        For the first kind, we are given an initial weight :math:`(W_0)_{i,j}` for each edge. This
        tells us how strong the connection between neurons :math:`i` and :math:`j` is immediately after a spike event.
        We then use the exponential decay function to model the decay of the connection strength over the 
        next :math:`c_w` time steps. This period is called the coupling window.
        Formally, at time step :math:`t` after a spike event, the weight of an edge between neurons :math:`i` and :math:`j` is given by

        .. math::
                W_{ij}(t) = \begin{cases}
                    (W_0)_{i,j} \: e^{-\beta t \Delta t} & \text{if } t < c_w \\
                    0 & \text{if } c_w \leq t
                \end{cases}

        The self-edges are used to implement the absolute and relative refractory periods. 
        A neuron enters the absolute refractory period after it spikes, during which it cannot spike again.
        The absolute refractory period is modelled by setting the weight of the self-edge to :math:`a`
        for :math:`A_{ref}` time steps. After this, the neuron enters the relative refractory period.
        During this period, the neuron can spike again but the probability of doing so is reduced.
        This is modelled by setting the weight of the self-edge to :math:`r e^{-\alpha t \Delta t}` for
        the next :math:`R_{ref}` time steps.
    
        That is, the weight of a self-edge at time t after a spike event is given by

        .. math::
                W_{ii}(t) = \begin{cases}
                    a & \text{if } t < A_{ref} \\
                    r e^{-\alpha t \Delta t} & \text{if } A_{ref} \leq t < A_{ref} + R_{ref} \\
                    0 & \text{if }  A_{ref} + R_{ref} \leq t
                \end{cases}

        All of this information can be represented by a tensor :math:`W` of shape :math:`N\times N\times T`, where
        :code:`W[i, j, t]` is the weight of the edge from neuron :math:`i` to neuron :math:`j` at time step :math:`t` after a spike event.

        Now, remove all the zero weights from :math:`W` and flatten the tensor to get a tensor of shape :math:`E\times T`, where
        :math:`E` is the number of edges in the network. Then, :code:`W[i, t]` is the weight of the :math:`i`-th edge at time step :math:`t` 
        after a spike event, and we can use the :code:`edge_index` tensor to tell us which edge corresponds to which neuron pair.

        A final remark: the weights are returned flipped in time to make the convolution operation more efficient.
        That is, :code:`W[i, T-t]` is the weight of the edge at time step :math:`t` after a spike event.

        Parameters
        -----------
        W0 : torch.Tensor [n_edges,]
            The initial weights of the edges
        edge_index : torch.Tensor [2, n_edges]
            The edge index

        Returns
        --------
        W : torch.Tensor [n_edges, T]
            The connectivity filter
        edge_index : torch.Tensor [2, n_edges]
            The edge index (with self edges added)
        """
        # Add self edges to the connectivity matrix
        n_edges = edge_index.shape[1]
        n_neurons = edge_index.max().item() + 1
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=n_neurons)
        W0 = torch.cat([W0, torch.zeros(edge_index.shape[1] - n_edges, device=W0.device)], dim=0)

        # Compute the indices of the self edges
        is_self_edge = edge_index[0] == edge_index[1]

        # Time steps going back T time steps
        t = torch.arange(self.T, device=self.T.device)

        # Compute the connectivity matrix
        abs_ref = self.abs_ref_strength * (t < self.abs_ref_scale) # The first abs_ref_scale time steps are absolute refractory
        rel_ref = (
                self.rel_ref_strength * torch.exp(-torch.abs(self.beta) * (t - self.abs_ref_scale)*self.dt)
                * (self.abs_ref_scale <= t) * (t <= self.abs_ref_scale + self.rel_ref_scale)
            ) # The next rel_ref_scale time steps are relative refractory and decay exponentially with decay rate beta
        refractory_edges = (abs_ref + rel_ref).repeat(is_self_edge.sum(), 1)
        coupling_edges = W0[~is_self_edge].unsqueeze(1) * torch.exp(-torch.abs(self.alpha)*self.dt * t)*(t < self.coupling_window)

        W = torch.zeros((W0.shape[0], self.T), device=self.T.device, dtype=torch.float32)
        W[is_self_edge] = refractory_edges
        W[~is_self_edge] = coupling_edges

        return W.flip(1), edge_index