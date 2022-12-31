import torch
import torch.nn as nn
from spiking_network.stimulation.base_stimulation import BaseStimulation

class SinStimulation(BaseStimulation):
    def __init__(self, targets: int | list, amplitudes: float | list, frequencies: float | list, durations: int | list, total_neurons: int, device='cpu'):
        super().__init__(targets, durations, total_neurons, device)

        if isinstance(amplitudes, (int, float)):
            amplitudes = [amplitudes] * self.n_targets
        if isinstance(frequencies, (int, float)):
            frequencies = [frequencies] * self.n_targets

        self.amplitudes = torch.tensor(amplitudes, device=device, dtype=torch.float32)
        self.frequencies = torch.tensor(frequencies, device=device, dtype=torch.float32)

        if any(self.amplitudes < 0):
            raise ValueError("All amplitudes must be positive.")
        if any(self.frequencies < 0):
            raise ValueError("All frequencies must be positive.")
        if any(self.frequencies > 1):
            raise ValueError("Period of sin stimulation must be more than one time step.")

        self._params = self._init_parameters(
            {
                "amplitudes": self.amplitudes,
                "frequencies": self.frequencies,
                "offsets": torch.zeros(self.n_targets, device=device, dtype=torch.float32)
            }
        )
        
    def __call__(self, t):
        if self.durations.max() <= t or t < 0:
            return torch.zeros(self.total_neurons, device=self.device)
        stimuli = self._params["amplitudes"] * torch.sin(2 * torch.pi * self._params['frequencies'] * t + self._params['offsets'])
        return self.distribute(stimuli)