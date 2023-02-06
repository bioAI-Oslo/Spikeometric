import torch
import torch.nn as nn

class PoissonStimulation(nn.Module):
    r"""
    Poisson stimulation of neurons.

    The stimulation times are modeled as a Poisson process with mean interval :math:`\lambda` and total duration :math:`T`.
    The stimulation events are of duration :math:`\tau` and strength :math:`s`.

    Parameters
    ----------
    strength : float
        Strength of the stimulation :math:`s`
    mean_interval : int
        Mean interval :math:`\lambda` between stimulation events. (ms)
    n_events : int
        Number of stimulation events.
    tau : int
        Duration of stimulation events. (ms)
    start : float
        Start time of the first stimulation event. (ms)
    rng : torch.Generator
        Random number generator.
    """
    def __init__(self, strength: float, mean_interval: int, n_events: int, tau: float, start: float = 0, rng=None):
        super().__init__()
        if tau < 0:
            raise ValueError("Stimulation length must be positive.")
        if mean_interval < 0:
            raise ValueError("Intervals must be positive.")
        
        self.register_buffer("tau", torch.tensor(tau, dtype=torch.int))
        self.register_buffer("mean_interval", torch.tensor(mean_interval, dtype=torch.int))
        self.register_buffer("n_events", torch.tensor(n_events, dtype=torch.int))
        self.register_buffer("start", torch.tensor(start, dtype=torch.float))

        self.register_parameter("strength", nn.Parameter(torch.tensor(strength, dtype=torch.float)))

        self._rng = rng if rng is not None else torch.Generator()

        self.requires_grad_(False)

        self.stimulation_times = self._generate_stimulation_plan(self.mean_interval, self.n_events)
    
    def _generate_stimulation_plan(self, mean_interval, n_events):
        """Generate poisson stimulation times.

        Parameters
        ----------
        intervals : torch.Tensor
            Mean interval between stimuli.
        n_events : torch.Tensor
            Number of stimulation events.

        Returns
        -------
        torch.Tensor
            Stimulation times with intervals drawn from a poisson distribution.
        """
        intervals = torch.poisson(mean_interval*torch.ones(n_events - 1), generator=self._rng)
        stimulus_times = torch.cumsum(
            torch.cat([torch.zeros(1), intervals]), dim=0
        )
        return stimulus_times + self.start

    def __call__(self, t):
        r"""
        Computes stimulation at time t. The stimulation is 0 if t is not in the interval of a stimulation event
        and :math:`s` if t is.

        Parameters
        ----------
        t : torch.Tensor
            Time at which to compute the stimulation.
        returns : torch.Tensor
            Stimulation at time t.
        """
        mask = (t - self.stimulation_times < self.tau) * (t - self.stimulation_times >= 0)
        return self.strength * mask.any()