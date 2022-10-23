import torch
from spiking_network.stimulation.abstract_stimulation import Stimulation

class SinStimulation(Stimulation):
    def __init__(self, targets, amplitudes, frequencies, duration, n_neurons, device='cpu'):
        super().__init__(targets, duration, n_neurons, device)
        amplitude = amplitudes if isinstance(amplitudes, list) else [amplitudes]*len(targets)
        frequency = frequencies if isinstance(frequencies, list) else [frequencies]*len(targets)

        self.amplitude = torch.tensor(amplitude, device=device)
        self.frequency = torch.tensor(frequency, device=device)
        self.duration = duration
        self.n_neurons = n_neurons

    def __call__(self, t):
        if t > self.duration:
            return torch.zeros(self.n_neurons, device=self.device)
        stimuli = self.amplitude * torch.sin(2 * torch.pi * self.frequency * t)
        return self.distribute(stimuli)
