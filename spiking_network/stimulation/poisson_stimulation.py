import torch

class PoissonStimulation:
    def __init__(self, targets, strength, t, dt=1):
        self.targets = targets
        self.strength = strength
        self.t = t
        self.dt = dt

    def __call__(self, shape):
        return torch.rand(shape) < self.rate * self.dt


def generate_poisson_stim_times(period, low, high, size, rng=None):
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
        isi += clipped_poisson(period, 100, low, high, rng=rng).tolist()
    cum = np.cumsum(isi)
    cum = cum[cum < size].astype(int)
    binned_stim_times = np.zeros(size)
    binned_stim_times[cum] = 1
    return np.expand_dims(binned_stim_times, 0)



def clipped_poisson(mu, size, low, high, max_iter=100000, rng=None):
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
