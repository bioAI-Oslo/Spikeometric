import torch
import torch.nn as nn
from spikeometric.stimulus.base_stimulus import BaseStimulus
import math
from typing import Union

class SinStimulus(BaseStimulus):
    r"""
    Sinusoidal stimulus of neurons.

    The stimulus is a sinusoidal function with amplitude :math:`A`, period :math:`T`, and phase :math:`\phi`.
    The stimulus starts at time :math:`t_0` and lasts for a duration :math:`\tau`.

    Parameters
    ----------
    amplitude : float
        Amplitude :math:`A` of stimulus.
    period : float
        Period :math:`T` of stimulus
    duration : int
        Duration :math:`\tau` stimulus in total
    stimulus_masks:
        A mask of shape (n_neurons,) indicating which neurons to stimulate.
    batch_size : int
        The number of networks to stimulate in parallel.
    phase : float
        Phase of stimulus :math:`\phi`
    baseline: float
        The constant baseline of the stimulus.
    start : float
        Start time :math:`t_0` of stimulus.
    dt : float
        Time step :math:`\Delta t` of the simulation in ms.
    """
    def __init__(self, amplitude: float, period: float, duration: int, stimulus_masks: torch.Tensor, batch_size: int = 1, phase: float = 0., baseline: float = 0, start: float = 0., dt: float = 1.):
        super().__init__()
        if amplitude < 0:
            raise ValueError("All amplitudes must be positive.")
        if period < 0:
            raise ValueError("All periods must be positive.")
        if duration < 0:
            raise ValueError("All durations must be positive.")
        
        self.register_parameter("amplitude", nn.Parameter(torch.tensor(amplitude, dtype=torch.float)))
        self.register_parameter("period", nn.Parameter(torch.tensor(period, dtype=torch.float)))
        self.register_parameter("phase", nn.Parameter(torch.tensor(phase, dtype=torch.float)))
        self.register_parameter("baseline", nn.Parameter(torch.tensor(baseline, dtype=torch.float)))
        self.register_buffer("duration", torch.tensor(duration, dtype=torch.int))
        self.register_buffer("start", torch.tensor(start, dtype=torch.float))
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))

        if isinstance(stimulus_masks, torch.Tensor) and stimulus_masks.ndim == 1:
            stimulus_masks = [stimulus_masks]

        if isinstance(stimulus_masks, torch.Tensor) and stimulus_masks.ndim == 2:
            stimulus_masks = [sm.squeeze() for sm in torch.split(stimulus_masks, 1, dim=0)]
        
        conc_stimulus_masks, split_points = self.batch_stimulus_masks(stimulus_masks, batch_size)
        self.n_batches = math.ceil(len(stimulus_masks) / batch_size)
        self.register_buffer("conc_stimulus_masks", conc_stimulus_masks)
        self.register_buffer("split_points", torch.tensor(split_points, dtype=torch.int))
        self._idx = 0

        self.requires_grad_(False)

    def __call__(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        r"""
        Computes stimulus at time t by applying a sinusoidal function.

        Between the start time :math:`t_0` and the end time :math:`t_0 + \tau`, the stimulus is given by

        .. math::
            f(t) = A \sin \left( \frac{2 \pi}{T} (t - t_0)\Delta t + \phi \right)
        """
        time_mask = (t < self.duration) * (t >= self.start)
        if torch.is_tensor(t):
            stim_mask = self.stimulus_masks[self._idx].unsqueeze(1)
            return (self.amplitude * torch.sin(2*torch.pi / self.period * (t-self.start)*self.dt + self.phase) + self.baseline) * time_mask * stim_mask
        return (
            self.amplitude * torch.sin(2*torch.pi / self.period * (t-self.start)*self.dt + self.phase) + self.baseline
        ) * time_mask * self.stimulus_masks[self._idx]