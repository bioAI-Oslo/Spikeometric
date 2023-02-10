from spiking_network.stimulation.base_stimulation import BaseStimulation
import torch

class MixedStimulation(BaseStimulation):
    def __init__(self, stimulations):
        self.targets = torch.cat([stim.targets for stim in stimulations])
        self.duration = max([stim.duration for stim in stimulations])
        self.n_neurons = stimulations[0].n_neurons
        self.device = stimulations[0].device
        self.stimulations = stimulations

    def __call__(self, t):
        """Return stimulus at time t."""
        return sum([stim(t) for stim in self.stimulations])

    def __dict__(self):
        return {
            "stimulation_type": "mixed",
            "targets": self.targets,
            "duration": self.duration,
            "n_neurons": self.n_neurons,
            "stimulations": [stim.__dict__() for stim in self.stimulations]
        }

