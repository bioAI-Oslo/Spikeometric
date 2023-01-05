import torch
import torch.nn as nn
from spiking_network.stimulation.base_stimulation import BaseStimulation

class RegularStimulation(BaseStimulation):
    def __init__(self, targets: int, intervals: int, strengths: float, temporal_scale: int, durations: int,  total_neurons: int, decays = 0.5, device="cpu"):
        super(RegularStimulation, self).__init__(targets, durations, total_neurons, device)
        self.temporal_scale = temporal_scale
        self.intervals = self._register_attribute(intervals, self.device)
        self.strengths = self._register_attribute(strengths, self.device).float()
        self.decays = self._register_attribute(decays, self.device).float()

        if temporal_scale < 0:
            raise ValueError("All temporal scales must be positive.")
        if any(self.intervals < 0):
            raise ValueError("All intervals must be positive.")
        if any(self.decays < 0):
            raise ValueError("All decays must be positive.")
        if not (len(self.intervals) == len(self.strengths) == len(self.decays) == self.n_targets):
            raise ValueError("All parameters must have the same length as targets.")

        self._params = self._init_parameters(
            {
                "strengths": self.strengths,
                "decays": self.decays,
            }
        )
        self._stimulation_times = self._get_stimulation_times(self.intervals, self.durations)
        self._stimulation_strengths = self._get_strengths(self._params)
    
    def _get_strengths(self, params: nn.ParameterDict) -> torch.Tensor:
        """Construct strength tensor from temporal_scale."""
        strengths = params["strengths"].unsqueeze(1).repeat(1, self.temporal_scale)
        decay_rates = -params.decays.unsqueeze(1).repeat(1, self.temporal_scale)
        time = torch.arange(self.temporal_scale, device=self.device).repeat(self.n_targets, 1).flip(1)
        decay = torch.exp(decay_rates*time)
        return strengths * decay

    def _get_stimulation_times(self, intervals, durations) -> torch.Tensor:
        """Generate regular stimulus onset times"""
        stim_times = torch.zeros((self.n_targets, self.durations.max()+self.temporal_scale - 1), device=self.device)
        stim_times[:, self.temporal_scale - 1:] = self._generate_stimulation_times(intervals, durations)
        return stim_times

    def stimulate(self, t):
        """Return stimulus at time t."""
        shifted_t = t + self.temporal_scale 
        stim_times = self._stimulation_times[:, t:shifted_t]
        strengths = self.stimulation_strengths
        return torch.sum(stim_times * strengths, axis=1)
    
    @property
    def stimulation_strengths(self):
        if self._params["strengths"].requires_grad:
            return self._get_strengths(self._params)
        return self._stimulation_strengths

    def _generate_stimulation_times(self, intervals, durations):
        """Generate regular stimulus times.
 
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
        n_intervals = torch.ceil(durations / intervals).int()
        max_n_intervals = n_intervals.max().item()
        intervals = intervals.unsqueeze(1).repeat(1, max_n_intervals)
        cum = torch.cumsum(intervals, dim=1)
        max_cum = cum.max().item()

        indices = torch.arange(self.n_targets, device=self.device).unsqueeze(1).repeat(1, max_n_intervals)
        stim_times = torch.stack((indices, cum), dim=2).transpose(0, 2).flatten(1)

        # Creates a tensor of shape (n_targets, max_duration) where the value at each index (i, t) is 1 if neueron i is stimulated at time t.
        binned_stim_times = torch.zeros((len(durations), max_cum+1), device=self.device)
        binned_stim_times[stim_times[0], stim_times[1]] = 1
        binned_stim_times = binned_stim_times[:, :durations.max()] # Trims the tensor to the correct size.

        # Remove the stimulus times that are outside the duration of the stimulus.
        duration_mask = durations.unsqueeze(1).repeat(1, durations.max()) < torch.arange(durations.max(), device=self.device).unsqueeze(0).repeat(len(durations), 1)
        binned_stim_times[duration_mask] = 0
        return binned_stim_times
