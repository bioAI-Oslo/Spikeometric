import torch
import torch.nn as nn
from spikeometric.stimulus.base_stimulus import BaseStimulus
import math
from typing import Union

class PoissonStimulus(BaseStimulus):
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
    stimulus_mask : torch.Tensor[bool]
        A mask of shape (n_neurons,) indicating which neurons to stimulate.
    batch_size : int
        Number of networks to stimulate in parallel.
    tau : int
        Duration of stimulus events
    dt: float
        Time step of the simulation in ms.
    start : float
        Start time of the first stimulus event. (ms)
    rng : torch.Generator
        Random number generator.
    """
    def __init__(self, strength: float, mean_interval: int, duration: int, stimulus_masks: torch.Tensor, batch_size: int = 1, tau: int = 1, dt: float = 1, start: float = 0, rng=None):
        super().__init__()
        if tau < 0:
            raise ValueError("Stimulus length must be positive.")
        if mean_interval < 0:
            raise ValueError("Intervals must be positive.")

        # Buffers
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))
        self.register_buffer("tau", torch.tensor(tau, dtype=torch.int))
        self.register_buffer("mean_interval", torch.tensor(mean_interval, dtype=torch.int))
        self.register_buffer("duration", torch.tensor(duration, dtype=torch.int))
        self.register_buffer("start", torch.tensor(start, dtype=torch.int))

        if isinstance(stimulus_masks, torch.Tensor) and stimulus_masks.ndim == 1:
            stimulus_masks = [stimulus_masks]

        if isinstance(stimulus_masks, torch.Tensor) and stimulus_masks.ndim == 2:
            stimulus_masks = [sm.squeeze() for sm in torch.split(stimulus_masks, 1, dim=0)]
        
        conc_stimulus_masks, split_points = self.batch_stimulus_masks(stimulus_masks, batch_size)
        self.register_buffer("split_points", torch.tensor(split_points, dtype=torch.int))
        self.register_buffer("conc_stimulus_masks", conc_stimulus_masks)

        self.n_batches = math.ceil(len(stimulus_masks) / batch_size)
        self._idx = 0

        self.register_parameter("strength", nn.Parameter(torch.tensor(strength, dtype=torch.float)))

        self._rng = rng if rng is not None else torch.Generator()

        self.requires_grad_(False)

        stimulus_times = self._generate_stimulus_plan(self.mean_interval, self.duration)
        self.register_buffer("stimulus_times", stimulus_times)
    
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

    def __call__(self, t: Union[torch.Tensor, float]) -> torch.Tensor:
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
        if torch.is_tensor(t):
            result = torch.zeros(self.stimulus_masks[self._idx].shape[0], t.shape[0], dtype=torch.float, device=self.stimulus_masks[self._idx].device)
            stimulus_times = self.stimulus_times.unsqueeze(1)
            mask = (t - stimulus_times < self.tau) * (t - stimulus_times >= 0)
            result[:, :self.duration] = self.strength * mask.any(0)[:self.duration] * self.stimulus_masks[self._idx].unsqueeze(1)
            return result

        mask = (t - self.stimulus_times < self.tau) * (t - self.stimulus_times >= 0)
        return self.strength * mask.any() * self.stimulus_masks[self._idx]