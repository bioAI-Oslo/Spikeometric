import torch
import torch.nn as nn
from spiking_network.stimulation.base_stimulation import BaseStimulation

class SinStimulation(BaseStimulation):
    def __init__(self, targets, amplitudes, frequencies, duration, n_neurons, device='cpu'):
        super().__init__(targets, duration, n_neurons, device)
        n_targets = len(targets) if isinstance(targets, list) else 1
        if isinstance(amplitudes, (int, float)):
            amplitudes = [amplitudes] * n_targets
        if isinstance(frequencies, (int, float)):
            frequencies = [frequencies] * n_targets

        self.amplitude = torch.tensor(amplitudes, device=device, dtype=torch.float)
        self.frequency = torch.tensor(frequencies, device=device)
        
        self.params = nn.ParameterDict({
            "amplitude": nn.Parameter(self.amplitude, requires_grad=True),
            "frequency": nn.Parameter(self.frequency, requires_grad=True),
            "offset": nn.Parameter(torch.zeros(n_targets, device=device), requires_grad=True),
        })
        
        self.duration = duration
        self.n_neurons = n_neurons

    def __call__(self, t):
        if t > self.duration:
            return torch.zeros(self.n_neurons, device=self.device)
        stimuli = self.params["amplitude"] * torch.sin(2 * torch.pi * self.params['frequency'] * t + self.params['offset'])
        return self.distribute(stimuli)

    
    def parameter_dict(self):
        return {
            "stimulation_type": "sin",
            "targets": self.targets,
            "duration": self.duration,
            "n_neurons": self.n_neurons,
            "amplitudes": self.amplitude,
            "frequencies": self.frequency
        }
    
