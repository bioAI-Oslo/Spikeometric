import torch
import numpy as np
from numpy.random import default_rng
from spiking_network.stimulation.base_stimulation import BaseStimulation

class PoissonStimulation(BaseStimulation):
    def __init__(self, targets, periods, temporal_scales, strengths, duration, n_neurons, seed=0, device='cpu'):
        super().__init__(targets, duration, n_neurons, device)
        self.periods = periods if isinstance(periods, list) else [periods]*len(self.targets)
        self.temporal_scales = temporal_scales if isinstance(temporal_scales, list) else [temporal_scales]*len(self.targets)
        self.strengths = strengths if isinstance(strengths, list) else [strengths]*len(self.targets)
        self.max_temporal_scale = max(self.temporal_scales)

        self.stimulation_times = self._get_stimulation_times(self.periods, duration, seed).to(device)
        self.stimulation_strengths = self._get_strengths(self.strengths, self.temporal_scales).to(device)

    def _get_stimulation_times(self, periods, duration, seed, low=1, high=10e3):
        """Generate Poisson stimulus onset times"""
        stim_times = []
        for period in periods:
            rng = default_rng(seed)
            stim_times.append(
                    torch.tensor(self.generate_poisson_stim_times(period, low, high, duration, rng=rng))
                )
            seed += 1
        return torch.cat(stim_times, dim=0)

    def _get_strengths(self, strengths, temporal_scales):
        """Construct strength tensor from temporal_scales."""
        max_temp_scale = max(temporal_scales)
        stimulation_strengths = torch.zeros((len(strengths), max_temp_scale))
        for i, (strength, temp_scale) in enumerate(zip(strengths, temporal_scales)):
            stimulation_strengths[i, :temp_scale] = strength
        return stimulation_strengths

    def __call__(self, t):
        """Return stimulus at time t."""
        if t > self.duration:
            return torch.zeros((self.n_neurons, 1), device=self.device)
        temp_scale = self.max_temporal_scale if t > self.max_temporal_scale else t
        stim_times = self.stimulation_times[:, t - temp_scale:t]
        strengths = self.stimulation_strengths[:, :temp_scale]
        stimuli = torch.sum(stim_times * strengths, axis=1)
        return self.distribute(stimuli)

    def clipped_poisson(self, mu, size, low, high, max_iter=100000, rng=None):
        """Generate Poisson distribution clipped between low and high.

        Parameters
        ----------
        mu : float
            Desired mean `mu`.
        size : int
            Number of samples `size`.
        low : float
            Lower bound `low`.
        high : flaot
            Upper bound `high`.
        max_iter : int
            Maximum number of iterations (the default is 100000).
        rng : generator
            Random number generator if None default is numpy default_rng
            (the default is None).

        Returns
        -------
        array
            Samples

        Examples
        --------
        >>> samples = clipped_poisson(10, 100, 10, 100)

        """
        rng = default_rng() if rng is None else rng
        truncated = rng.poisson(mu, size=size)
        itr = 0
        while ((truncated < low) | (truncated > high)).any():
            mask, = np.where((truncated < low) | (truncated > high))
            temp = rng.poisson(mu, size=size)
            temp_mask, = np.where((temp >= low) & (temp <= high))
            mask = mask[:len(temp_mask)]
            truncated[mask] = temp[:len(mask)]
            itr += 1
            if itr > max_iter:
                print('Did not reach the desired limits in "max_iter" iterations')
                return None
        return truncated


    def generate_poisson_stim_times(self, period, low, high, size, rng=None):
        """Generate poisson stimulus times.

        Parameters
        ----------
        period : float
            Mean period between stimulus onsets.
        low : float
            Lower cutoff.
        high : float
            Upper cutoff.
        size : int
            Number of time steps.
        rng : generator
            Random number generator if None default is numpy default_rng
            (the default is None).

        Returns
        -------
        array
            Stimulus times.

        Examples
        --------
        >>> rng = default_rng(1234)
        >>> generate_poisson_stim_times(2, 2, 4, 10, rng=rng)
        array([[0., 0., 0., 1., 0., 0., 0., 1., 0., 1.]])
        """
        isi = []
        while sum(isi) < size:
            isi += self.clipped_poisson(period, 100, low, high, rng=rng).tolist()
        cum = np.cumsum(isi)
        cum = cum[cum < size].astype(int)
        binned_stim_times = np.zeros(size)
        binned_stim_times[cum] = 1
        return np.expand_dims(binned_stim_times, 0)

    def __dict__(self):
        return {
            "stimulation_type": "poisson",
            'targets': self.targets,
            'periods': self.periods,
            'temporal_scales': self.temporal_scales,
            'strengths': self.strengths,
            'duration': self.duration,
            'n_neurons': self.n_neurons,
        }

if __name__ == '__main__':
    targets = [0, 5]
    periods = [10, 4]
    temporal_scales = [20, 3]
    strengths = [1, 2]
    duration = 100
    stim = PoissonStimulation(targets, periods, temporal_scales, strengths, duration, seed=0)
