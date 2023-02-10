import torch
import torch.nn as nn

class RegularStimulus(nn.Module):
    r"""
    Regular stimulus of neurons.

    The first stimulus event starts at time :math:`t_0` and the following events are spaced by intervals :math:`\Delta t`.
    The stimulus events last for a duration :math:`\tau` and have a strength :math:`s`.

    Parameters
    ----------
    strength : float
        Strength :math:`s` of the stimulus.
    start : float
        Start time :math:`t_0` of the first stimulus event. (ms)
    interval : float
        Interval :math:`\Delta t` between that beginning of each stimulus events. (ms)
    n_events : int
        Number of stimulus events.
    tau : float
        Duration :math:`\tau` of each stimulus event. (ms)
    """
    def __init__(self, strength: float, interval: float, n_events: int, tau: float, start: float=0.):
        super(RegularStimulus, self).__init__()
        if tau < 0:
            raise ValueError("Temporal scale must be positive.")
        if interval < 0:
            raise ValueError("Interval of stimulus must be positive.")

        self.register_buffer("start", torch.tensor(start, dtype=torch.float))
        self.register_buffer("interval", torch.tensor(interval, dtype=torch.float))
        self.register_buffer("n_events", torch.tensor(n_events, dtype=torch.int))
        self.register_buffer("tau", torch.tensor(tau, dtype=torch.float))

        self.register_parameter("strength", nn.Parameter(torch.tensor(strength, dtype=torch.float)))

        self.requires_grad_(False)

    def __call__(self, t):
        r"""
        Computes the stimulus at time :math:`t`. The stimulus is constant at :math:`s` during
        the intervals :math:`[t_i, t_i + n\tau]` and zero otherwise.

        Parameters
        ----------
        t : torch.Tensor
            Time :math:`t` at which to compute the stimulus (ms).

        Returns
        -------
        torch.Tensor
            Stimulus at time :math:`t`.
        """
        return self.strength*(t % self.interval < self.tau)*(t >= self.start)*(t < self.start + self.n_events*self.interval)