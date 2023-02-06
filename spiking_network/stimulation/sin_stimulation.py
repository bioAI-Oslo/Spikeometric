import torch
import torch.nn as nn

class SinStimulation(nn.Module):
    r"""
    Sinusoidal stimulation of neurons.

    The stimulation is a sinusoidal function with amplitude :math:`A`, period :math:`T`, and phase :math:`\phi`.
    The stimulation starts at time :math:`t_0` and lasts for a duration :math:`\tau`.

    Parameters
    ----------
    amplitude : float
        Amplitude :math:`A` of stimulation.
    period : float
        Period :math:`T` of stimulation (ms).
    duration : int
        Duration :math:`\tau` stimulation in total. (ms)
    phase : float
        Phase of stimulation :math:`\phi`. (ms)
    d: float
        The constant baseline of the stimulation.
    start : float
        Start time :math:`t_0` of stimulation. (ms)
    """
    def __init__(self, amplitude: float, period: float, duration: int, phase: float = 0., d: float = 0, start: float = 0.):
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
        self.register_parameter("d", nn.Parameter(torch.tensor(d, dtype=torch.float)))
        self.register_buffer("duration", torch.tensor(duration, dtype=torch.int))
        self.register_buffer("start", torch.tensor(start, dtype=torch.float))
        self.requires_grad_(False)

    def __call__(self, t):
        r"""
        Computes stimulation at time t by applying a sinusoidal function.

        Between the start time :math:`t_0` and the end time :math:`t_0 + \tau`, the stimulation is given by

        .. math::
            f(t) = A \sin \left( \frac{2 \pi}{T} (t - t_0) + \phi \right)
        """
        return (
            self.amplitude * torch.sin(2*torch.pi / self.period * (t-self.start) + self.phase) + self.d
        ) * (t < self.duration) * (t >= self.start)