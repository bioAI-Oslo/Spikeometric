import torch
import torch.nn as nn

class PoissonStimulus(nn.Module):
    r"""
    Poisson stimulus of neurons.

    The stimulus times are modeled as a Poisson process with mean interval :math:`\lambda` and total duration :math:`T`.
    The stimulus events are of duration :math:`\tau` and strength :math:`s`.

    Parameters
    ----------
    strength : float
        Strength of the stimulus :math:`s`
    mean_interval : int
        Mean interval :math:`\lambda` between stimulus events
    duration : int
        Total duration :math:`T` of the stimulus.
    tau : int
        Duration of stimulus events
    dt: float
        Time step of the simulation in ms.
    start : float
        Start time of the first stimulus event. (ms)
    rng : torch.Generator
        Random number generator.
    """
    def __init__(self, strength: float, mean_interval: int, duration: int, tau: float, dt: float = 1, start: float = 0, rng=None):
        super().__init__()
        if tau < 0:
            raise ValueError("Stimulus length must be positive.")
        if mean_interval < 0:
            raise ValueError("Intervals must be positive.")

        # Buffers
        self.register_buffer("dt", torch.tensor(1, dtype=torch.float))
        self.register_buffer("tau", torch.tensor(tau, dtype=torch.int))
        self.register_buffer("mean_interval", torch.tensor(mean_interval, dtype=torch.int))
        self.register_buffer("duration", torch.tensor(duration, dtype=torch.int))
        self.register_buffer("start", torch.tensor(start, dtype=torch.int))

        self.register_parameter("strength", nn.Parameter(torch.tensor(strength, dtype=torch.float)))

        self._rng = rng if rng is not None else torch.Generator()

        self.requires_grad_(False)

        self.stimulus_times = self._generate_stimulus_plan(self.mean_interval, self.duration)
    
    def _generate_stimulus_plan(self, mean_interval, duration):
        """Generate poisson stimulus times.

        Parameters
        ----------
        intervals : torch.Tensor
            Mean interval between stimuli.
        duration : torch.Tensor
            Duration of the stimulus.

        Returns
        -------
        torch.Tensor
            Stimulus times with intervals drawn from a poisson distribution.
        """
        intervals = torch.poisson(mean_interval*torch.ones(duration), generator=self._rng)
        stimulus_times = torch.cumsum(
            torch.cat([torch.zeros(1), intervals]), dim=0
        )
        return stimulus_times[stimulus_times < duration] + self.start

    def __call__(self, t):
        r"""
        Computes stimulus at time t. The stimulus is 0 if t is not in the interval of a stimulus event
        and :math:`s` if t is.

        Parameters
        ----------
        t : torch.Tensor
            Time at which to compute the stimulus.
        returns : torch.Tensor
            Stimulus at time t.
        """
        mask = (t - self.stimulus_times < self.tau) * (t - self.stimulus_times >= 0)
        return self.strength * mask.any()