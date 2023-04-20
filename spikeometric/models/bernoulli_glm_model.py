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

    More formally, the model can be broken into three steps, each of which is implemented as a separate method in this class:

        #. .. math:: g_i(t+1) = \sum_{\tau=0}^{T-1} \left(X_i(t-\tau)r(\tau) + \sum_{j \in \mathcal{N}(i)} (W_0)_{j, i} X_j(t-\tau) c(\tau)\right) + \mathcal{E}_i(t+1)
        #. .. math:: p_i(t+1) = \sigma(g_i(t+1) - \theta) \Delta t
        #. .. math:: X_i(t+1) \sim \text{Bernoulli}(p_i(t+1))

    The first equation is implemented in the :meth:`input` method and gives us the input to the neuron :math:`i` at time :math:`t+1` as a sum of the refractory, synaptic and external inputs.
    The refractory input is calculated by convolving the spike history of the neuron itself with a refractory filter :math:`r`, the synaptic input is obtained by convolving the spike history 
    of the neuron's neighbors with the coupling filter :math:`c`, weighted by the synaptic weights :math:`W_0`, and the exteral input is given by evaluating an external input function :math:`\mathcal{E}` at time :math:`t+1`.

    The second equation is implemented in :meth:`non_linearity` which computes the probability that the neuron :math:`i` spikes at time :math:`t+1` by passing 
    its input :math:`g_i(t+1)` through a sigmoid nonlinearity with threshold :math:`\theta`. The probability is then scaled by the time step size :math:`\Delta t` to get the probability of spiking
    in an interval of length :math:`\Delta t`.

    Finally, the third equation is implemented in :meth:`emit_spikes` which samples the spike of the neuron :math:`i` at time :math:`t+1` from the Bernoulli distribution with probability :math:`p_i(t+1)`.

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
        T = max(coupling_window, abs_ref_scale + rel_ref_scale) # The total number of time steps we need consider for the refractory and coupling filters

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
            g_i(t+1) = \sum_{\tau=0}^{T-1} \left(X_i(t-\tau)r(\tau) + \sum_{j \in \mathcal{N}(i)} (W_0)_{j, i} X_j(t-\tau) c(\tau)\right) + \mathcal{E}_i(t+1)

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
            p_i(t+1) = \sigma(g_i(t+1) - \theta)

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
            P(X_i(t+1) = 1) = p_i(t+1)

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
        This is done by filtering the initial coupling weights :math:`W_0` with the coupling filter :math:`c` 
        and using a refractory filter :math:`r` as self-edge weights to emulate the refractory period.

        For the coupling edges, we are given an initial weight :math:`(W_0)_{i,j}` for each edge. This
        tells us how strong the connection between neurons :math:`i` and :math:`j` is immediately after a spike event.
        We then use an exponential decay as our coupling filter :math:`c` to model the decay of the connection strength over the 
        next :math:`c_w` time steps. This period is called the coupling window.
        Formally, at time step :math:`t` after a spike event, the weight of an edge between neurons :math:`i` and :math:`j` is given by

        .. math::
                W_{i, j}(t) = \begin{cases}
                    (W_0)_{i,j} \: e^{-\beta t \Delta t} & \text{if } t < c_w \\
                    0 & \text{if } c_w \leq t
                \end{cases}

        The self-edges are used to implement the absolute and relative refractory periods. 
        A neuron enters an absolute refractory period after it spikes, during which it cannot spike again.
        The absolute refractory period is modeled by setting the weight of the self-edge to :math:`a`
        for :math:`A_{ref}` time steps. After this, the neuron enters the relative refractory period.
        During this period, the neuron can spike again but the probability of doing so is reduced.
        This is modeled by weighting spike events by to :math:`r e^{-\alpha t \Delta t}` for
        the next :math:`R_{ref}` time steps.
    
        That is, the refractory filter :math:`r` is given by

        .. math::
                r(t) = \begin{cases}
                    a & \text{if } t < A_{ref} \\
                    r e^{-\alpha t \Delta t} & \text{if } A_{ref} \leq t < A_{ref} + R_{ref} \\
                    0 & \text{if }  A_{ref} + R_{ref} \leq t
                \end{cases}

        And we set `W_{i, i}(t) = r(t)` for all neurons :math:`i`.

        All of this information can be represented by a tensor :math:`W` of shape :math:`N\times N\times T`, where
        :code:`W[i, j, t]` is the weight of the edge from neuron :math:`i` to neuron :math:`j` at time step :math:`t` after a spike event.

        Now, we remove all the zero weights from :math:`W` and flatten the tensor to get a tensor of shape :math:`E\times T`, where
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
        n_edges = W0.shape[0]
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