from spikeometric.models.base_model import BaseModel
import torch
from tqdm import tqdm

class ExponentialGLM(BaseModel):
    r"""
    The Exponential GLM model from section S.7 of the paper 
    `"Systematic errors in connectivity inferred from activity in strongly coupled recurrent circuits"
    <https://www.biorxiv.org/content/10.1101/512053v1.full>`_.

    It is a Poisson GLM model that passes the input to each neuron through an exponential nonlinearity
    and samples a spike count from a Poisson distribution.

    More specifically, at time :math:`t+1`, the input of the network is calculated from the state at time :math:`t` as follows:

    .. math::
        g_i(t+1) = r \: \sum_j \mathbf{W_{j,i}} \cdot \mathbf{x_j} + b_i + f_i(t+1)

    where :math:`r` is the scaling of the recurrent connections, :math:`b_i` is the strength of a uniform background input
    and :math:`f_i(t+1)` is the stimulus input of neuron :math:`i` at time :math:`t+1`.

    The response to the input is then passed through an exponential nonlinearity:

    .. math::
        \mu_i(t+1) = \frac{1}{\alpha}\: e^{\beta g_i(t+1)}

    where :math:`\alpha` and :math:`\beta` are parameters of the model that control the nonlinearity.

    The spikes are then sampled from a Poisson distribution with rate :math:`\mu_i(t+1)`:

    .. math::
        x_i(t+1) = \text{Pois}(\mu_i(t+1))
    
    Parameters
    ----------
    alpha : float
        The :math:`\alpha` parameter of the model. (tunable)
    beta : float
        The :math:`\beta` parameter of the model. (tunable)
    T : int
        The number of milliseconds to consider back in time for the exponential filter.
    tau : float
        The time constant of the exponential filter.
    dt : float
        The time step of the simulation.
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
        T = int(T/dt)
        self.register_buffer("T", torch.tensor(T, dtype=torch.int))
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float32))

        # Parameters
        self.register_parameter("alpha", torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float32)))
        self.register_parameter("beta", torch.nn.Parameter(torch.tensor(beta, dtype=torch.float32)))
        self.register_parameter("tau", torch.nn.Parameter(torch.tensor(tau, dtype=torch.float32)))
        self.register_parameter("r", torch.nn.Parameter(torch.tensor(r, dtype=torch.float32)))
        self.register_parameter("b", torch.nn.Parameter(torch.tensor(b, dtype=torch.float32)))

        # RNG
        self._rng = rng if rng is not None else torch.Generator()

        self.requires_grad_(False)

    def input(self, edge_index: torch.Tensor, W: torch.Tensor, state: torch.Tensor, t=-1, stimulus_mask: torch.Tensor = False) -> torch.Tensor:
        r"""
        The input to the network at time t+1.

        .. math::
            g_i(t+1) = r \sum_j \mathbf{W_{j,i}} \cdot \mathbf{x_j} + b_i + f_i(t+1)

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

        .. math::
            \mu_i(t+1) = \frac{1}{\alpha} e^{\beta g_i(t+1)}

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

        .. math::
            x_i(t+1) = \text{Pois}(\mu_i(t+1))

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
        The connectivity filter of the network. This is used to calculate the recurrent input.

        The synaptic weights are filtered with an exponential decay, so that
        the weight of the connection between two neurons :math:`i` and :math:`j` at time :math:`t-T` 
        before the current time step is:

        .. math::
            W_{ij}(t) = (W_0)_{ij} \: e^{(t - T) \Delta t / \tau} 

        Spikes that are emmited more than :math:`T` time steps ago have no effect on the input.

        Parameters
        ----------
        W0 : torch.Tensor[float]
            The initial synaptic weights of the network.
        edge_index : torch.Tensor[int]
            The edge index of the network.
        
        Returns
        --------
        torch.Tensor
            The connectivity filter of the network.
        """
        t = torch.arange(1, self.T+1, dtype=torch.float32, device=W0.device).repeat(W0.shape[0], 1)
        return torch.einsum("i, ij -> ij", W0, torch.exp((t -self.T)*self.dt/self.tau))