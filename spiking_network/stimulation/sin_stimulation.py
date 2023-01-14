import torch
import torch.nn as nn
from spiking_network.stimulation.base_stimulation import BaseStimulation

class SinStimulation(BaseStimulation):
    def __init__(self, amplitude: float, frequency: float, duration: int, device='cpu'):
        super().__init__(device)
        self.amplitude = torch.tensor(amplitude, device=device, dtype=torch.float32)
        self.frequency = torch.tensor(frequency, device=device, dtype=torch.float32)
        self.duration = duration

        if self.amplitude < 0:
            raise ValueError("All amplitudes must be positive.")
        if self.frequency < 0:
            raise ValueError("All frequencies must be positive.")
        if self.frequency > 1:
            raise ValueError("Period of sin stimulation must be more than one time step.")
        if self.duration < 0:
            raise ValueError("All durations must be positive.")

        self._tunable_params = self._init_parameters(
            {
                "amplitude": self.amplitude,
                "frequency": self.frequency,
                "offset": torch.tensor(0, device=device, dtype=torch.float32),
            }
        )

    def stimulate(self, t):
        return self._tunable_params["amplitude"] * torch.sin(2 * torch.pi * self._tunable_params['frequency'] * t + self._tunable_params['offset'])