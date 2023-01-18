import torch
import torch.nn as nn
from spiking_network.stimulation.base_stimulation import BaseStimulation

class RegularStimulation(BaseStimulation):
    def __init__(self, interval: int, strength: float, duration: int, temporal_scale=1, decay = 0.5, device="cpu"):
        super(RegularStimulation, self).__init__()
        if duration < 0:
            raise ValueError("All durations must be positive.")
        if temporal_scale < 0:
            raise ValueError("Temporal scale must be positive.")
        if interval < 0:
            raise ValueError("Interval of stimulation must be positive.")
        if decay < 0:
            raise ValueError("Decay must be positive.")

        self.register_buffer("temporal_scale", torch.tensor(temporal_scale))
        self.register_buffer("interval", torch.tensor(interval))
        self.register_buffer("duration", torch.tensor(duration))
        self.register_parameter("strength", nn.Parameter(torch.tensor(strength)))
        self.register_parameter("decay", nn.Parameter(torch.tensor(decay)))
        self.requires_grad_(False)

        self._stimulation_times = self._get_stimulation_times(self.interval, self.duration, self.temporal_scale)
        self._stimulation_strengths = self._get_strengths(self.strength, self.duration, self.temporal_scale)
    
    def _get_strengths(self, strength, decay, temporal_scale) -> torch.Tensor:
        """Construct strength tensor from temporal_scale."""
        strengths = strength * torch.ones(temporal_scale)
        decay_rates = -decay * torch.ones(temporal_scale)
        time = torch.arange(temporal_scale).flip(0)
        decay = torch.exp(decay_rates*time)
        return strengths * decay

    def _get_stimulation_times(self, interval, duration, temporal_scale) -> torch.Tensor:
        """Generate regular stimulus onset times"""
        stim_times = torch.zeros(duration + temporal_scale-1)
        stim_times[torch.arange(temporal_scale - 1, duration+temporal_scale-1, interval)] = 1
        return stim_times

    def stimulate(self, t):
        """Return stimulus at time t."""
        shifted_t = t + self.temporal_scale 
        stim_times = self._stimulation_times[t:shifted_t]
        return torch.sum(stim_times * self.stimulation_strengths)

    @property
    def stimulation_strengths(self):
        if self.strength.requires_grad:
            return self._get_strengths(self.strength, self.decay, self.temporal_scale)
        return self._stimulation_strengths