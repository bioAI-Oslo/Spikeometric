import torch
import torch.nn as nn
from spikeometric.stimulus.base_stimulus import BaseStimulus
import math
from typing import Union

class RegularStimulus(BaseStimulus):
    r"""
    Regular stimulus of neurons with period of :math:`T` time steps and a duration :math:`\tau` time steps for each stimulus event.
    The stimulus is constant at :math:`s` during the stimulus events and zero otherwise.
    The first stimulus event starts at time step :math:`t_0` and the stimulus ends at time step :math:`t_s`.

    Parameters
    ----------
    strength : float
        Strength :math:`s` of the stimulus.
    period : int
        period :math:`T` of the stimulus
    tau : int
        Duration :math:`\tau` of each stimulus event
    stop : int
        Stop time :math:`t_s` of the stimulus
    stimulus_mask : torch.Tensor[bool]
        A mask of shape (n_neurons,) indicating which neurons to stimulate.
    batch_size : int
        Number of networks to stimulate in parallel.
    start : int
        Start time :math:`t_0` of the first stimulus event
    dt : float
        Time step :math:`\Delta t` of the simulation in ms.
    """
    def __init__(self, strength: float, period: int, tau: int, stop: int, stimulus_masks: torch.Tensor, batch_size: int = 1, start: int=0, dt: float=1.):
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

        if isinstance(stimulus_masks, torch.Tensor) and stimulus_masks.ndim == 1:
            stimulus_masks = [stimulus_masks]

        if isinstance(stimulus_masks, torch.Tensor) and stimulus_masks.ndim == 2:
            stimulus_masks = [sm.squeeze() for sm in torch.split(stimulus_masks, 1, dim=0)]
        
        conc_stimulus_masks, split_points = self.batch_stimulus_masks(stimulus_masks, batch_size)
        self.register_buffer("conc_stimulus_masks", conc_stimulus_masks)
        self.register_buffer("split_points", torch.tensor(split_points, dtype=torch.int))

        self.n_batches = math.ceil(len(stimulus_masks) / batch_size)
        self._idx = 0

        self.register_parameter("strength", nn.Parameter(torch.tensor(strength, dtype=torch.float)))

        self.requires_grad_(False)

    def __call__(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
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
        if torch.is_tensor(t):
            stim_mask = self.stimulus_masks[self._idx].unsqueeze(1)
            return self.strength*(t % self.period < self.tau)*(t >= self.start)*(t < self.stop) * stim_mask
        return self.strength*(t % self.period < self.tau)*(t >= self.start)*(t < self.stop) * self.stimulus_masks[self._idx]