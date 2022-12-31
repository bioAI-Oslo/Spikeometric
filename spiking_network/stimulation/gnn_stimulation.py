from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
import numpy
from spiking_network.stimulation.base_stimulation import BaseStimulation

class GNNStimulation(BaseStimulation):
    def __init__(self, targets: torch.Tensor, intervals: torch.Tensor, strengths: torch.Tensor, temporal_scales: torch.Tensor, durations: torch.Tensor, total_neurons: int, decay = 0.5, device="cpu"):
        super(GNNStimulation, self).__init__(targets, durations, total_neurons, device)
        # convert the parameters to match targets
        if len(intervals.shape) == 0:
            intervals = intervals.unsqueeze(0)
            intervals = intervals.repeat(self.n_targets)
        if len(strengths.shape) == 0:
            strengths = strengths.unsqueeze(0)
            strengths = strengths.repeat(self.n_targets)
        if len(temporal_scales.shape) == 0:
            temporal_scales = temporal_scales.unsqueeze(0)
            temporal_scales = temporal_scales.repeat(self.n_targets)

        self.intervals = intervals
        self.strengths = strengths
        self.temporal_scales = temporal_scales

        self.source_node = self.total_neurons
        self.edge_index = torch.tensor([[self.source_node] * self.n_targets, self.targets], dtype=torch.long)
        self.decay = torch.tensor(decay, device=device)

        self._params = self._init_parameters({"intervals": self.intervals, "strengths": self.strengths, "temporal_scales": self.temporal_scales, "durations": self.durations, "decay": self.decay}, device)

        self._max_temporal_scale = self.temporal_scales.max().item()
        self._stimulation_times = self._get_stimulation_times(self._params)

    def _get_strengths(self, params: nn.ParameterDict) -> torch.Tensor:
        """Construct strength tensor from temporal_scales."""
        strengths = params["strengths"].unsqueeze(1).repeat(1, self._max_temporal_scale)
        decay = torch.exp(-params["decay"]*torch.arange(self._max_temporal_scale, device=self.device)).unsqueeze(0).flip(1)
        return strengths * decay
    
    def _get_stimulation_times(self, params: nn.ParameterDict) -> torch.Tensor:
        """Generate regular stimulus onset times"""
        stim_times = torch.zeros((self.total_neurons+1, self.durations.max()+self._max_temporal_scale - 1))
        for i, interval in enumerate(params["intervals"]):
            stim_times[self.targets[i], torch.arange(self._max_temporal_scale - 1, self.durations[i] + self._max_temporal_scale - 1, interval)] = 1
        return stim_times

    def forward(self, t):
        if self.durations.max() <= t or t < 0:
            return torch.zeros((self.n_targets, 1), device=self.device)
        shifted_t = t + self._max_temporal_scale
        onset = self._stimulation_times[:, t:shifted_t]
        return self.propagate(self.edge_index, onset=onset, strength=self._get_stimulation_strengths()).squeeze()

    def message(self, onset_i, strength):
        return torch.sum(onset_i * strength, axis=1).unsqueeze(1)

    def _untunable_parameters(self):
        return ["intervals", "temporal_scales", "durations"]
