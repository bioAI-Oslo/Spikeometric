import torch

class RegularStimulation:
    def __init__(self, targets, rate, strength, time):
        self.stimulation_times = self._generate_stim_times(rate, time)
        self.stimulation_strength = strength * torch.ones(len(targets))
        self.edge_index = self._generate_edge_index(targets)

    def _generate_stim_times(self, rate, time):
        x = torch.zeros(time)
        x[torch.arange(0, time, rate)] = 1
        return x

    def _generate_edge_index(self, targets):
        edge_index = torch.zeros(2, len(targets), dtype=torch.long)
        edge_index[1, :] = torch.tensor(targets)
        return edge_index
