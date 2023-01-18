import torch
import torch.nn as nn
from spiking_network.stimulation.base_stimulation import BaseStimulation

class SinStimulation(BaseStimulation):
    def __init__(self, amplitude: float, frequency: float, duration: int):
        super().__init__()
        if amplitude < 0:
            raise ValueError("All amplitudes must be positive.")
        if frequency < 0:
            raise ValueError("All frequencies must be positive.")
        if frequency > 1:
            raise ValueError("Period of sin stimulation must be more than one time step.")
        if duration < 0:
            raise ValueError("All durations must be positive.")
        
        self.register_parameter("amplitude", nn.Parameter(torch.tensor(amplitude)))
        self.register_parameter("frequency", nn.Parameter(torch.tensor(frequency)))
        self.register_parameter("offset", nn.Parameter(torch.tensor(0.)))
        self.register_buffer("duration", torch.tensor(duration))
        self.requires_grad_(False)


    def stimulate(self, t):
        return self.amplitude * torch.sin(2 * torch.pi * self.frequency * t + self.offset)