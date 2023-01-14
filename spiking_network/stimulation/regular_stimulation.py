import torch
import torch.nn as nn
from spiking_network.stimulation.base_stimulation import BaseStimulation

class RegularStimulation(BaseStimulation):
    def __init__(self, interval: int, strength: float, duration: int, temporal_scale=1, decay = 0.5, device="cpu"):
        super(RegularStimulation, self).__init__(device)
        if duration < 0:
            raise ValueError("All durations must be positive.")
        if temporal_scale < 0:
            raise ValueError("Temporal scale must be positive.")
        if interval < 0:
            raise ValueError("Interval of stimulation must be positive.")
        if decay < 0:
            raise ValueError("Decay must be positive.")
        
        self.temporal_scale = temporal_scale
        self.interval = interval
        self.duration = duration

        self.strength = torch.tensor(strength, device=device, dtype=torch.float)
        self.decay = torch.tensor(decay, device=device, dtype=torch.float)

        self._tunable_params = self._init_parameters(
            {
                k: v for k, v in self.__dict__.items() if k in self._tunable_parameters
            }
        )

        self._stimulation_times = self._get_stimulation_times(interval, duration, temporal_scale)
        self._stimulation_strengths = self._get_strengths(self._tunable_params)
    
    def _get_strengths(self, params: nn.ParameterDict) -> torch.Tensor:
        """Construct strength tensor from temporal_scale."""
        strengths = params.strength * torch.ones(self.temporal_scale, device=self.device)
        decay_rates = -params.decay * torch.ones(self.temporal_scale, device=self.device)
        time = torch.arange(self.temporal_scale, device=self.device).flip(0)
        decay = torch.exp(decay_rates*time)
        return strengths * decay

    def _get_stimulation_times(self, interval, duration, temporal_scale) -> torch.Tensor:
        """Generate regular stimulus onset times"""
        stim_times = torch.zeros(duration + temporal_scale-1, device=self.device)
        stim_times[torch.arange(temporal_scale - 1, duration+temporal_scale-1, interval, device=self.device)] = 1
        return stim_times

    def stimulate(self, t):
        """Return stimulus at time t."""
        shifted_t = t + self.temporal_scale 
        stim_times = self._stimulation_times[t:shifted_t]
        return torch.sum(stim_times * self.stimulation_strengths)

    @property
    def stimulation_strengths(self):
        if self._tunable_params.strength.requires_grad:
            return self._get_strengths(self._tunable_params)
        return self._stimulation_strengths

    @property
    def _tunable_parameters(self):
        return {"strength", "decay"}

    @property
    def _parameter_keys(self):
        return {"interval", "temporal_scale", "duration", "strength", "decay"}