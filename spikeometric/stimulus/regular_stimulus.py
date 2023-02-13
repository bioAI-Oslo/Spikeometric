import torch
import torch.nn as nn

class RegularStimulus(nn.Module):
    r"""
    Regular stimulus of neurons with period of :math:`T` time steps and a duration :math:`\tau` time steps for each stimulus event.
    The stimulus is constant at :math:`s` during the stimulus events and zero otherwise.
    The first stimulus event starts at time step :math:`t_0` and the stimulus ends at time step :math:`t_s`.

    Parameters
    ----------
    strength : float
        Strength :math:`s` of the stimulus.
    start : int
        Start time :math:`t_0` of the first stimulus event
    period : int
        period :math:`T` of the stimulus
    stop : int
        Stop time :math:`t_s` of the stimulus
    tau : int
        Duration :math:`\tau` of each stimulus event
    dt : float
        Time step :math:`\Delta t` of the simulation in ms.
    """
    def __init__(self, strength: float, period: int, tau: int, stop: int, start: int=0, dt: float=1.):
        super(RegularStimulus, self).__init__()
        if tau < 0:
            raise ValueError("Temporal scale must be positive.")
        if period < 0:
            raise ValueError("Period of stimulus must be positive.")

        self.register_buffer("start", torch.tensor(start, dtype=torch.int))
        self.register_buffer("period", torch.tensor(period, dtype=torch.int))
        self.register_buffer("stop", torch.tensor(stop, dtype=torch.int))
        self.register_buffer("tau", torch.tensor(tau, dtype=torch.float))
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))

        self.register_parameter("strength", nn.Parameter(torch.tensor(strength, dtype=torch.float)))

        self.requires_grad_(False)

    def __call__(self, t):
        r"""
        Computes the stimulus at time step :math:`t`. The stimulus is constant at :math:`s` during
        the stimulus events and zero otherwise.

        Parameters
        ----------
        t : torch.Tensor
            Time :math:`t` at which to compute the stimulus (ms).

        Returns
        -------
        torch.Tensor
            Stimulus at time :math:`t`.
        """
        return self.strength*(t % self.period < self.tau)*(t >= self.start)*(t < self.stop)