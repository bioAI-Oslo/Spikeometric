import torch
import torch.nn as nn
import numpy
from spiking_network.stimulation.base_stimulation import BaseStimulation

class RegularStimulation(BaseStimulation):
    def __init__(self, targets, rates, strengths, temporal_scales, duration, n_neurons, device="cpu"):
        super(RegularStimulation, self).__init__(targets, duration, n_neurons, device)
        # convert the parameters to tensors
        n_targets = len(targets) if isinstance(targets, list) else 1
        if isinstance(rates, (int, float)):
            rates = [rates] * n_targets
        if isinstance(strengths, (int, float)):
            strengths = [strengths] * n_targets
        if isinstance(temporal_scales, (int, float)):
            temporal_scales = [temporal_scales] * n_targets

        self.strengths = torch.tensor(strengths, device=device, dtype=torch.float)
        self.rates = torch.tensor(rates, device=device)
        self.temporal_scales = torch.tensor(temporal_scales, device=device)
        self.max_temporal_scale = self.temporal_scales.max().item()

        self.params = nn.ParameterDict({
            "strengths": nn.Parameter(self.strengths, requires_grad=True),
            "rates": nn.Parameter(self.rates, requires_grad=False),
            "temporal_scales": nn.Parameter(self.temporal_scales, requires_grad=False)
        })

        self.stimulation_strengths = self._get_strengths(self.params)
        self.stimulation_times = self._get_stimulation_times(self.params, duration)

    def _get_strengths(self, params):
        """Construct strength tensor from temporal_scales."""
        strengths = params["strengths"].unsqueeze(1).repeat(1, self.max_temporal_scale)
        return strengths

    def _get_stimulation_times(self, params, duration):
        """Generate regular stimulus onset times"""
        stim_times = torch.zeros((len(self.targets), duration))
        for i, rate in enumerate(params["rates"]):
            stim_times[i, torch.arange(0, duration, int(1/rate))] = 1
        return stim_times

    def __call__(self, t):
        if self.duration < t:
            return torch.zeros((self.n_neurons,), device=self.device)
        temp_scale = self.max_temporal_scale if self.max_temporal_scale < t else t
        stim_times = self.stimulation_times[:, t - temp_scale:t]
        strengths = self.stimulation_strengths[:, :temp_scale]
        stimuli = torch.sum(stim_times * strengths, axis=1)
        return self.distribute(stimuli)

    
    def parameter_dict(self):
        return {
            "stimulation_type": "regular",
            "targets": self.targets,
            "duration": self.duration,
            "n_neurons": self.n_neurons,
            "strengths": self.strengths,
            "rates": self.rates,
            "temporal_scales": self.temporal_scales
        }
    

if __name__ == '__main__':
    targets = [0, 5]
    periods = [0.5, 0.1]
    temporal_scales = [20, 3]
    strengths = [1, 2]
    duration = 100
    stim = RegularStimulation(targets, periods, strengths, temporal_scales, duration)
