import torch
import torch.nn as nn
import numpy
from spiking_network.stimulation.base_stimulation import BaseStimulation

class RegularStimulation(BaseStimulation):
    def __init__(self, targets: int | list, intervals: int | list, strengths: float | list, temporal_scales: int | list, durations: int | list,  total_neurons: int, decay = 0.5, device="cpu"):
        super(RegularStimulation, self).__init__(targets, durations, total_neurons, device)
        # convert the parameters to tensors
        if isinstance(intervals, (int, float)):
            intervals = [intervals] * self.n_targets
        if isinstance(strengths, (int, float)):
            strengths = [strengths] * self.n_targets
        if isinstance(temporal_scales, int):
            temporal_scales = [temporal_scales] * self.n_targets
 
        self.strengths = torch.tensor(strengths, device=device, dtype=torch.float32)
        self.intervals = torch.tensor(intervals, device=device)
        self.temporal_scales = torch.tensor(temporal_scales, device=device)
        self.decay = torch.tensor(decay, device=device)
        
        # Validate the parameters
        if any(self.intervals < 0):
            raise ValueError("All intervals must be positive.")
        if not (len(self.targets) == len(self.strengths) == len(self.intervals) == len(self.temporal_scales)):
            raise ValueError("All parameters must have the same length.")

        self._params = self._init_parameters(
            {
                "strengths": self.strengths,
                "decay": self.decay,
            }
        )
        
        self._max_temporal_scale = self.temporal_scales.max().item()
        self._stimulation_times = self._get_stimulation_times(self._params)

    def _get_strengths(self, params: nn.ParameterDict) -> torch.Tensor:
        """Construct strength tensor from temporal_scales."""
        strengths = params["strengths"].unsqueeze(1).repeat(1, self._max_temporal_scale)
        decay = torch.exp(-params["decay"]*torch.arange(self._max_temporal_scale, device=self.device)).unsqueeze(0).flip(1)
        return strengths * decay

    def _get_stimulation_times(self, params: nn.ParameterDict) -> torch.Tensor:
        """Generate regular stimulus onset times"""
        stim_times = torch.zeros((len(self.targets), self.durations.max()+self._max_temporal_scale - 1))
        for i, interval in enumerate(self.intervals):
            stim_times[i, torch.arange(self._max_temporal_scale - 1, self.durations[i] + self._max_temporal_scale - 1, interval)] = 1
        return stim_times

    def __call__(self, t: int) -> torch.Tensor:
        """Return stimulus at time t."""
        if self.durations.max() <= t or t < 0:
            return torch.zeros(self.total_neurons, device=self.device)
        shifted_t = t + self._max_temporal_scale
        stim_times = self._stimulation_times[:, t:shifted_t]
        strengths = self._get_strengths(self._params)
        stimuli = torch.sum(stim_times * strengths, axis=1)
        return self.distribute(stimuli)