from spikeometric.models.sa_model import SAModel
import torch
from tqdm import tqdm

class RectifiedSAM(SAModel):
    r"""
    The Rectified Synaptic Activation Model from section S.5 of the paper `"Systematic errors in connectivity inferred from activity in strongly coupled recurrent circuits" <https://www.biorxiv.org/content/10.1101/512053v1.full>`_.

    This is a Linear-Nonlinear-Poisson model that uses a rectified linear non-linearity and Poisson spiking. 
    It can be descibed as a mixture of the ThresholdSAM and the RectifiedLNP models. It uses activation as 
    state like the ThresholdSAM, but uses a rectified non-linearity and Poisson spiking like the RectifiedLNP.
    Unlike the ThresholdSAM, the RectifiedSAM is tunable.

    It is defined by the following equations:

        #. .. math:: g_i(t+1) = r \: \sum_j (W_0)_{j,i}\: s_j(t) + b_i + \mathcal{E}_i(t+1)
        #. .. math:: \mu_i(t+1) = \lambda_0\Delta t[g_i(t+1)) - \theta]_+
        #. .. math:: X_i(t+1) \sim \text{Pois}(\mu_i(t+1))
    
    The first equation is implemented in the :meth:`input` method. It computes the input to neuron :math:`i` at time :math:`t+1`
    by adding the synaptic input from neighbouring neurons, weighted by the connectivity matrix :math:`W_0` and scaled by :math:`r`,
    along with the background input :math:`b_i` and the stimulus input :math:`\mathcal{E}_i(t+1)`.

    The second equation is implemented in the :meth:`non_linearity` method. It computes the expected firing rate of neuron :math:`i` at time :math:`t+1`
    by applying a rectified linear non-linearity to the input.

    The third equation is implemented in the :meth:`emit_spikes` method. It generates a spike from a Poisson process with rate :math:`\mu_i(t+1)`.

    Between each time step, the activation is decayed decayed by a factor of :math:`(1 - \Delta t/\tau)` 
    if no spikes are emitted and incremented by :math:`\Delta t` otherwise.

    Parameters
    ----------
    lambda_0 : float
        The scaling of the rectified non-linearity.
    theta : float
        The threshold of the rectified non-linearity.
    tau : float
        The time constant of the neurons.
    dt : float
        The time step of the simulation.
    r : float
        The scaling of the recurrent connections.
    b : float
        The baseline strength of the background input.
    rng : torch.Generator, optional
        The random number generator to use for the Poisson process.
    """
    def __init__(self, lambda_0: float, theta: float, tau: float, dt: float, r: float, b: float, rng=None):
        super().__init__()
        # Buffers
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))
        self.register_buffer("T", torch.tensor(1, dtype=torch.int))

        # Parameters
        self.register_parameter("lambda_0", torch.nn.Parameter(torch.tensor(lambda_0, dtype=torch.float)))
        self.register_parameter("theta", torch.nn.Parameter(torch.tensor(theta, dtype=torch.float)))
        self.register_parameter("tau", torch.nn.Parameter(torch.tensor(tau, dtype=torch.float)))
        self.register_parameter("r", torch.nn.Parameter(torch.tensor(r, dtype=torch.float)))
        self.register_parameter("b", torch.nn.Parameter(torch.tensor(b, dtype=torch.float)))

        # RNG
        self._rng = rng if rng is not None else torch.Generator()

        self.requires_grad_(False)

    def input(self, edge_index: torch.Tensor, W: torch.Tensor, state: torch.Tensor, t=-1, stimulus_mask=False) -> torch.Tensor:
        r"""
        The input to the network at time t+1.

        .. math:: g_i(t+1) = r \: \sum_j (W_0)_{j,i}\: s_j(t) + b_i + \mathcal{E}_i(t+1)

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

        .. math:: \mu_i(t+1) = \lambda_0\Delta t[g_i(t+1)) - \theta]_+ 

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
        return torch.poisson(rates, generator=self._rng).to(dtype=torch.uint8)

    def update_activation(self, spikes: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        r"""
        Update the activation of the neurons according to the formula:

        .. math::
            s_i(t+1) = s_i(t)(1 - \frac{\Delta t}{\tau}) + X_i(t)\Delta t

        where :math:`s_i(t)` is the activation of neuron :math:`i` at time :math:`t`, :math:`X_i(t)`
        indicates whether the neuron spiked at time t or not, and :math:`\tau` is the time constant of the neurons.

        Parameters
        -----------
        spikes : torch.Tensor [n_neurons, 1]
            The spikes of the network at the current time step.
        activation : torch.Tensor [n_neurons, 1]
            The activation of the neurons at the previous time step.

        Returns
        --------
        activation : torch.Tensor [n_neurons, 1]
            The activation of the neurons at the current time step.
        """
        return activation*(1 - self.dt/self.tau) + spikes*self.dt

