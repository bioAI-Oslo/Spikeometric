from spikeometric.models.base_model import BaseModel
import torch

class RectifiedLNP(BaseModel):
    r"""
    The Rectified LNP model from section S.6 of the paper `"Systematic errors in connectivity inferred from activity in strongly coupled recurrent circuits" <https://www.biorxiv.org/content/10.1101/512053v1.full>`_.

    It is a Linear-Nonlinear-Poisson model which passes the input to each neuron through a rectified linear nonlinearity
    to give an expected firing rate and then samples from a Poisson distribution with that rate.

    More formally, the model is defined by the three equations:

        #. .. math:: g_i(t+1) = r \: \sum_{\tau = 0}^{T-1} \sum_{j\in \mathcal{N}(i)} (W_0)_{j, i} X_j(t-\tau)c(\tau) + b_i + \mathcal{E}_i(t+1)
        #. .. math:: \mu_i(t+1) = \lambda_0 \Delta t [g_i(t+1)) - \theta]_+ 
        #. .. math:: X_i(t+1) \sim \text{Pois}(\mu_i(t+1))

    The first equation is implemented by the :meth:`input` method and gives the input to neuron :math:`i` at time :math:`t+1`
    as a sum of a recurrent synaptic input, a uniform background input :math:`b_i` and an external input :math:`\mathcal{E}_i(t+1)`.
    The synaptic input is obtained by convolving the spike history of neighbouring neurons with a coupling filter :math:`c`, weighted by the
    synaptic weights :math:`W_0`. The strength of the recurrence is controlled by the parameter :math:`r`.

    The second equation is implemented by the :meth:`non_linearity` method and gives the expected firing rate of neuron :math:`i` at time :math:`t+1`
    by rectifying the thresholded input and scaling it by the parameter :math:`\lambda_0` and the time step :math:`\Delta t`.

    The third equation is implemented by the :meth:`emit_spikes` method and samples the spikes from a Poisson distribution with rate
    :math:`\mu_i(t+1)`.

    Parameters
    ----------
    lambda_0 : float
        The scaling of the response :math:`\lambda_0`
    theta : float
        The threshold :math:`\theta` of the rectified linear nonlinearity
    T : int
        The coupling window :math:`T` in time steps
    tau : float
        The time constant :math:`\tau` in the exponential filter in milliseconds
    dt : float
        The time step :math:`\Delta t`
    r : float
        The scaling of the recurrent connections :math:`r`
    b : float
        The strength of the uniform background input :math:`b`
    rng : torch.Generator, optional
        The random number generator to use for sampling the spikes, by default None
    """
    def __init__(self, lambda_0: float, theta: float, T: int, tau: float, dt: float, r: float, b: float, rng=None):
        super().__init__()
        # Buffers
        self.register_buffer("T", torch.tensor(T, dtype=torch.int))
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float32))

        # Parameters
        self.register_parameter("lambda_0", torch.nn.Parameter(torch.tensor(lambda_0, dtype=torch.float32)))
        self.register_parameter("theta", torch.nn.Parameter(torch.tensor(theta, dtype=torch.float32)))
        self.register_parameter("tau", torch.nn.Parameter(torch.tensor(tau, dtype=torch.float32)))
        self.register_parameter("r", torch.nn.Parameter(torch.tensor(r, dtype=torch.float32)))
        self.register_parameter("b", torch.nn.Parameter(torch.tensor(b, dtype=torch.float32)))

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
        Computes the response to the input through a rectified linear nonlinearity:

        .. math:: \mu_i(t+1) = \lambda_0\Delta t [g_i(t+1)) - \theta]_+ 

        Parameters
        ----------
        input : torch.Tensor
            The input to the network at time :math:`t+1`
        
        Returns
        --------
        torch.Tensor
            The response to the input at time :math:`t+1`
        """
        return self.lambda_0*torch.relu(input - self.theta) * self.dt

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
        return torch.poisson(rates, generator=self._rng)
    
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
            The edge index of the network
        """
        t = torch.arange(1, self.T+1, dtype=torch.float32, device=W0.device).repeat(W0.shape[0], 1)
        return torch.einsum("i, ij -> ij", W0, torch.exp((t -self.T)*self.dt/self.tau)), edge_index