from spikeometric.models.base_model import BaseModel
import torch

class PoissonGLM(BaseModel):
    r"""
    The Poisson GLM model from section S.7 of the paper 
    `"Systematic errors in connectivity inferred from activity in strongly coupled recurrent circuits"
    <https://www.biorxiv.org/content/10.1101/512053v1.full>`_.

    It is a Poisson Generalized Linear Model model that passes the input to each neuron through an exponential non-linearity
    and samples a spike count from a Poisson distribution.

    More specifically, we have the following equations:

        #. .. math:: g_i(t+1) = r \: \sum_{\tau = 0}^{T-1} \sum_{j\in \mathcal{N}(i)} (W_0)_{j, i} X_j(t-\tau)c(\tau) + b_i + \mathcal{E}_i(t+1)
        #. .. math:: \mu_i(t+1) = \frac{\Delta t}{\alpha}\: e^{\beta g_i(t+1)}
        #. .. math:: X_i(t+1) \sim \text{Pois}(\mu_i(t+1))

    The first equation is implemented in the :meth:`input` method and gives the input to neuron :math:`i` at time :math:`t+1`
    as a convolution of the spike history of the neighboring neurons with a coupling filter :math:`c`, weighted by the
    connectivity matrix :math:`W_0`, and scaled by the recurrent scaling factor :math:`r`. There is also a uniform background
    input :math:`b_i` and an external stimulus :math:`\mathcal{E}_i(t+1)`.

    The second equation is implemented in the :meth:`non_linearity` method and gives the mean spike count of neuron :math:`i`
    at time :math:`t+1` as a function of the input :math:`g_i(t+1)`. 

    Finally, the third equation is implemented in the :meth:`emit_spikes` method and samples the spike count of neuron :math:`i`
    at time :math:`t+1` from a Poisson distribution with mean :math:`\mu_i(t+1)`.

    
    Parameters
    ----------
    alpha : float
        The :math:`\alpha` parameter of the model. (tunable)
    beta : float
        The :math:`\beta` parameter of the model. (tunable)
    T : int
        The number of time steps to consider back in time.
    tau : float
        The time constant of the exponential filter.
    dt : float
        The time step of the simulation in milliseconds.
    r : float
        The scaling of the recurrent connections. (tunable)
    b : float
        The strength of the uniform background input. (tunable)
    rng : torch.Generator, optional
        The random number generator to use for sampling the spikes. If not provided, a new one will be created.
    """
    def __init__(self, alpha: float, beta: float, T: int, tau: float, dt: float, r: float, b: float, rng=None):
        super().__init__()
        # Buffers
        self.register_buffer("T", torch.tensor(T, dtype=torch.int))
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))

        # Parameters
        self.register_parameter("alpha", torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float)))
        self.register_parameter("beta", torch.nn.Parameter(torch.tensor(beta, dtype=torch.float)))
        self.register_parameter("tau", torch.nn.Parameter(torch.tensor(tau, dtype=torch.float)))
        self.register_parameter("r", torch.nn.Parameter(torch.tensor(r, dtype=torch.float)))
        self.register_parameter("b", torch.nn.Parameter(torch.tensor(b, dtype=torch.float)))

        # RNG
        self._rng = rng if rng is not None else torch.Generator()

        self.requires_grad_(False)

    def input(self, edge_index: torch.Tensor, W: torch.Tensor, state: torch.Tensor, t=-1, stimulus_mask: torch.Tensor = False) -> torch.Tensor:
        r"""
        The input to the network at time t+1.

        .. math:: g_i(t+1) = r \: \sum_{\tau = 0}^{T-1} \sum_{j\in \mathcal{N}(i)} (W_0)_{j, i} X_j(t-\tau)c(\tau) + b_i + \mathcal{E}_i(t+1)

        Parameters
        ----------
        edge_index : torch.Tensor[int]
            The edge index of the network.
        W : torch.Tensor[float]
            The weights of the network.
        state : torch.Tensor[int]
            The state of the network at time t.
        t : int
            The time step of the simulation.
        stimulus_mask : torch.Tensor[bool]
            A boolean mask of the neurons that are stimulated at time t+1.

        Returns
        -------
        torch.Tensor
            The input to the network at time t+1.
        """
        return (
            self.r*self.synaptic_input(edge_index, W, state)
             + self.stimulus_input(t, stimulus_mask)
             + self.b
        )
    
    def non_linearity(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        The exponential non-linearity of the model. Calculates an expected spike count from the input.

        .. math:: \mu_i(t+1) = \frac{\Delta t}{\alpha}\: e^{\beta g_i(t+1)}

        Parameters
        ----------
        input : torch.Tensor[float]
            The input to the network at time t+1.

        Returns
        --------
        torch.Tensor
            The expected spike count of the network at time t+1.
        """
        return 1/self.alpha * torch.exp(self.beta*input) * self.dt

    def emit_spikes(self, rates: torch.Tensor) -> torch.Tensor:
        r"""
        Samples the spikes from a Poisson distribution with rate :math:`\mu_i(t+1)`.

        .. math:: X_i(t+1) \sim \text{Pois}(\mu_i(t+1))

        Parameters
        ----------
        rates : torch.Tensor[float]
            The expected spike count of the network at time t+1.
        
        Returns
        --------
        torch.Tensor
            The state of the network at time t+1.
        """
        return torch.poisson(rates, generator=self._rng)*self.dt
    
    def connectivity_filter(self, W0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        r"""
        The connectivity filter of the network is a tensor that contains the synaptic weights
        between two neurons :math:`i` and :math:`j` at time step :math:`t` after a spike event.
        This is computed by filtering the initial synaptic weights :math:`W_0` with the
        exponetial coupling kernel :math:`c`:

        .. math::
            W_{i,j}(t) = (W_0)_{i,j} \: c(t) = (W_0)_{i,j} e^{- \Delta t \frac{t}{\tau}}

        Spikes that are emited more than :math:`T` time steps ago have no effect on the input.

        Parameters
        ----------
        W0 : torch.Tensor[float]
            The initial synaptic weights of the network.
        edge_index : torch.Tensor[int]
            The edge index of the network.
        
        Returns
        --------
        W : torch.Tensor[float]
            The connectivity filter of the network.
        edge_index : torch.Tensor[int]
            The edge index of the network.
        """
        t = torch.arange(1, self.T+1, dtype=torch.float32, device=W0.device).repeat(W0.shape[0], 1)
        return torch.einsum("i, ij -> ij", W0, torch.exp((t -self.T)*self.dt/self.tau)), edge_index