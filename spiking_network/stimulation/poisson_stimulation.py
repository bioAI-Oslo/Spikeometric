import torch
import torch.nn as nn
from spiking_network.stimulation.base_stimulation import BaseStimulation

class PoissonStimulation(BaseStimulation):
    def __init__(self, strength, interval, duration, temporal_scale=1, decay=0.2, seed=None, device='cpu'):
        super().__init__(device)
        self.temporal_scale = temporal_scale
        self.interval = interval
        self.strength = torch.tensor(strength, device=device, dtype=torch.float32)
        self.decay = torch.tensor(decay, device=device, dtype=torch.float32)
        self.duration = duration

        if temporal_scale < 0:
            raise ValueError("Temporal scale must be positive.")
        if self.interval < 0:
            raise ValueError("Intervals must be positive.")
        if self.decay < 0:
            raise ValueError("Decay must be positive.")

        self._tunable_params = self._init_parameters(
            {
                "strength": self.strength,
                "decay": self.decay,
            }
        )

        if seed is None:
            self._rng = torch.Generator()
        else:
            self._rng = torch.Generator().manual_seed(seed)

        self._stimulation_times = self._get_stimulation_times(self.interval, self.duration, self.temporal_scale)
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
        stim_times = torch.zeros((duration+temporal_scale - 1,), device=self.device)
        stim_times[temporal_scale - 1:] = self._generate_stimulation_times(interval, duration)
        return stim_times

    @property
    def stimulation_strengths(self):
        if self._tunable_params.strength.requires_grad:
            return self._get_strengths(self._tunable_params)
        return self._stimulation_strengths

    def stimulate(self, t):
        """Return stimulus at time t."""
        shifted_t = t + self.temporal_scale 
        stim_times = self._stimulation_times[t:shifted_t]
        return torch.sum(stim_times * self.stimulation_strengths)
    
    def _generate_stimulation_times(self, interval, duration):
        """Generate poisson stimulus times.

        Parameters
        ----------
        intervals : torch.Tensor
            Mean interval between stimuli.
        durations : torch.Tensor
            Duration of each stimulus.

        Returns
        -------
        tensor
            Samples
        """
        intervals = torch.ones((duration), device=self.device) * interval
        isi = torch.poisson(intervals, generator=self._rng)

        cum = torch.cumsum(isi, dim=0).to(dtype=torch.long)

        stim_times = torch.zeros((duration,), device=self.device)
        stim_times[cum[cum < duration]] = 1

        return stim_times