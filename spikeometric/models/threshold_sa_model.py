from spikeometric.models.sa_model import SAModel
import torch

class ThresholdSAM(SAModel):
    r"""
    The generative model used in the paper `"Systematic errors in connectivity inferred from activity in strongly coupled recurrent circuits" <https://www.biorxiv.org/content/10.1101/512053v1.full>`_.

    Is is a threshold-based model, where the neurons spike if the synaptic activation is above a threshold.
    Since the model is threshold-based, it is not tunable.

    It is defined by the following equations:

        #. .. math:: g_i(t+1) = r \: \sum_j (W_0)_{j,i}\: s_j(t) + b_i + \mathcal{E}_i(t+1)
        #. .. math:: X_i(t+1) = \begin{cases}
                                    1 & \text{if } g_i(t+1) > \theta \\
                                    0 & \text{otherwise}
                                \end{cases}

    where :math:`r` is the scaling of the recurrent connections, :math:`b_i` is the strength of a background input
    and :math:`\mathcal{E}_i(t+1)` is an external stimulus. The response to the input is then compared to a threshold :math:`\theta` 
    and the neurons spike if the activation is above the threshold.

    Between each time step, the synaptic activation is decayed by a factor of :math:`(1-\Delta t/\tau)` 
    if the neuron did not spike and increased by :math:`\Delta t` if it did.

    
    Parameters
    ----------
    r : float
        The scaling of the recurrent connections. (tunable)
    b : float
        The strength of the background input. (tunable)
    tau : float
        The time constant of the neurons.
    dt : float
        The time step of the simulation.
    sigma : float
        The standard deviation of the noise.
    rho : float
        The sparsity of the noise.
    theta : float
        The threshold of the neurons (the neurons will spike if the activation is above the threshold).
    rng : torch.Generator
        The random number generator.
    """

    def __init__(self, r: float, b: float, tau: float, dt: float, sigma: float, rho: float, theta: float, rng: torch.Generator):
        super().__init__()
        # Buffers
        self.register_buffer("r", torch.tensor(r, dtype=torch.float))
        self.register_buffer("b", torch.tensor(b, dtype=torch.float))
        self.register_buffer("tau", torch.tensor(tau, dtype=torch.float))
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float))
        self.register_buffer("rho", torch.tensor(rho, dtype=torch.float))
        self.register_buffer("theta", torch.tensor(theta, dtype=torch.float))
        self.register_buffer("T", torch.tensor(1, dtype=torch.int))
        self._rng = rng
    
    def input(self, edge_index: torch.Tensor, W: torch.Tensor, state: torch.Tensor, t=-1, stimulus_mask=False) -> torch.Tensor:
        r"""
        Calculates the input to each neuron as:

        .. math:: g_i(t+1) = r \: \sum_j (W_0)_{j,i}\: s_j(t) + b_i + \mathcal{E}_i(t+1)

        Parameters
        -----------
        edge_index : torch.Tensor [2, n_edges]
            The connectivity of the network.
        W : torch.Tensor [n_edges, 1]
            The edge weights of the connectivity filter.
        state : torch.Tensor [n_neurons, 1]
            The activation of the neurons at time t.
        t : int
            The current time step.
        stimulus_mask : torch.Tensor [n_neurons]
            A boolean mask of the neurons that are stimulated at time t.
        
        Returns
        --------
        input : torch.Tensor [n_neurons]
            The input to each neuron at time t.
        """
        return self.r*self.synaptic_input(edge_index, W, state=state) + self.background_input(state.shape[0]) + self.stimulus_input(t, stimulus_mask)

    def background_input(self, n_neurons: int):
        r"""
        Generate the background input. This is a uniform excitatory input given by the parameter :math:`b`
        plus a scaled Gaussian noise with standard deviation :math:`\sigma` and some sparsity.

        Parameters
        ------------
        t : int
            The current time step.
        n_neurons : int
            The number of neurons in the network.

        Returns
        --------
        background_input : torch.Tensor [n_edges]
        """
        noise = torch.normal(0., self.sigma, size=(n_neurons,), device=self.sigma.device, generator=self._rng)
        filtered = torch.rand(n_neurons, device=self.rho.device, generator=self._rng) < self.rho
        filtered_noise = noise * filtered
        return self.b*(1 + noise*filtered_noise)

    def emit_spikes(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Emit spikes from the network. The neuron will spike if the input it receives is above the threshold.

        .. math:: X_i(t+1) = \begin{cases}
                                    1 & \text{if } g_i(t+1) > \theta \\
                                    0 & \text{otherwise}
                                \end{cases}

        Parameters
        -----------
        activation : torch.Tensor [n_neurons]
            The activation of the neurons at the current time step.

        Returns
        --------
        spikes : torch.Tensor [n_neurons]
            The spikes of the network at the current time step.
        """
        return input > self.theta

    def update_activation(self, spikes: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        r"""
        Update the activation of the neurons according to the formula:

        .. math::
            s_i(t+1) = s_i(t)(1 - \frac{\Delta t}{\tau}) + X_i(t)\Delta t

        where :math:`s_i(t)` is the activation of neuron :math:`i` at time :math:`t`, :math:`X_i(t)`
        indicates whether the neuron spiked at time :math:`t` or not, and :math:`\tau` is the time constant of the neurons.

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