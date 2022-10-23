import torch
import numpy
from spiking_network.stimulation.abstract_stimulation import Stimulation

class RegularStimulation(Stimulation):
    def __init__(self, targets, rates, strengths, temporal_scales, duration, n_neurons, device="cpu"):
        super(RegularStimulation, self).__init__(targets, duration, n_neurons, device)
        rates = rates if isinstance(rates, list) else [rates]*len(targets)
        strengths = strengths if isinstance(strengths, list) else [strengths]*len(targets)
        temporal_scales = temporal_scales if isinstance(temporal_scales, list) else [temporal_scales]*len(targets)
        self.max_temporal_scale = max(temporal_scales)

        self.strengths = self._get_strengths(strengths, temporal_scales).to(device)
        self.stimulation_times = self._get_stimulation_times(rates, duration).to(device)

    def _get_strengths(self, strengths, temporal_scales):
        """Construct strength tensor from temporal_scales."""
        max_temp_scale = max(temporal_scales)
        strength_tensor = torch.zeros((len(strengths), max_temp_scale))
        for i, (strength, temp_scale) in enumerate(zip(strengths, temporal_scales)):
            strength_tensor[i, :temp_scale] = strength
        return strength_tensor

    def _get_stimulation_times(self, rates, duration):
        """Generate regular stimulus onset times"""
        stim_times = []
        for rate in rates:
            stim_times.append(
                self._regular_stimulation_times(rate, duration)
            )
        return torch.cat(stim_times, dim=0)

    def _regular_stimulation_times(self, rate, duration):
        """Generate regular stimulus onset times"""
        stim_times = torch.zeros(duration)
        stim_times[torch.arange(0, duration, int(1/rate))] = 1
        return stim_times.unsqueeze(0)

    def __call__(self, t):
        if self.duration < t:
            return torch.zeros((self.n_neurons, 1), device=self.device)
        temp_scale = self.max_temporal_scale if self.max_temporal_scale < t else t
        stim_times = self.stimulation_times[:, t - temp_scale:t]
        strengths = self.strengths[:, :temp_scale]
        stimuli = torch.sum(stim_times * strengths, axis=1)
        return self.distribute(stimuli)

if __name__ == '__main__':
    targets = [0, 5]
    periods = [0.5, 0.1]
    temporal_scales = [20, 3]
    strengths = [1, 2]
    duration = 100
    stim = RegularStimulation(targets, periods, strengths, temporal_scales, duration)
