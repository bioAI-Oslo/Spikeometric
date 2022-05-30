"""
This module contain all simulator and generator code

"""
import scipy.stats as st
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
from functools import partial
import pathlib


def clipped_lognormal(mu, sigma, size, low, high, rng=None, max_iter=100000):
    """Generate lognormal distribution clipped between low and high.

    Parameters
    ----------
    mu : float
        Desired mean `mu`.
    sigma : float
        Desired spread `sigma`.
    size : int
        number of samples `size`.
    low : float
        Lower cutoff value `low`.
    high : float
        Higher cutoff value `high`.
    rng : generator
        Random number generator if None numpy default_rng is used
        (the default is None).
    max_iter : int
        Maximum number of iterations `max_iter` (the default is 100000).

    Returns
    -------
    array
        Samples.

    Examples
    --------
    >>> samples = clipped_lognormal(10, 5, 100, 10, 200)

    """
    rng = default_rng() if rng is None else rng
    sample = rng.lognormal(mu, sigma, size)
    itr = 0
    while ((sample < low) | (sample > high)).any():
        mask = list(np.where((sample < low) | (sample > high)))
        subsample = rng.lognormal(mu, sigma, size)
        submask = list(np.where((subsample > low) & (subsample < high)))
        n = min(len(mask[0]), len(submask[0]))
        for i in range(len(mask)):
            mask[i] = mask[i][:n]
            submask[i] = submask[i][:n]
        sample[tuple(mask)] = subsample[tuple(submask)]
        itr += 1
        if itr > max_iter:
            print(f'Did not reach the desired limits in max_iter={max_iter} iterations')
            return None
    return sample


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


def construct_connectivity_matrix(params, rng=None, self_connections=False):
    """Construct connectivity matrix.

    Parameters
    ----------
    params : dict
        Must contain 'uniform', 'normal' or 'glorot_normal' `params`.
    rng : generator
        Random number generator if None default is numpy default_rng
        (the default is None).
    self_connections : bool
        Allow self connections or not (the default is False).

    Returns
    -------
    ndarray
        Connectivity matrix.

    Examples
    --------
    >>> params = {'normal': {'mu': 0, 'sigma': 1}, 'n_neurons': 10}
    >>> W_0 = construct_connectivity_matrix(params)

    """
    rng = default_rng() if rng is None else rng
    if 'uniform' in params:
        W_0 = rng.uniform(
            low=params['uniform']['low'],
            high=params['uniform']['high'],
            size=(params['n_neurons'], params['n_neurons'])
        )
    elif 'normal' in params:
        W_0 = rng.normal(
            loc=params['normal']['mu'],
            scale=params['normal']['sigma'],
            size=(params['n_neurons'], params['n_neurons'])
        )
    elif 'glorot_normal' in params:
        W_0 = rng.normal(
            loc=params['glorot_normal']['mu'],
            scale=params['glorot_normal']['sigma'] / np.sqrt(params['n_neurons']),
            size=(params['n_neurons'], params['n_neurons'])
        )
    elif 'lognormal' in params:
        W_ex = clipped_lognormal(
            mu=params['lognormal']['mu_ex'],
            sigma=params['lognormal']['sigma_ex'],
            size=(params['n_neurons_ex'], params['n_neurons']),
            low=params['lognormal']['low_ex'],
            high=params['lognormal']['high_ex'],
        )
        W_in = clipped_lognormal(
            mu=params['lognormal']['mu_in'],
            sigma=params['lognormal']['sigma_in'],
            size=(params['n_neurons_in'], params['n_neurons']),
            low=params['lognormal']['low_in'],
            high=params['lognormal']['high_in'],
        )
        W_ex = sparsify(W_ex, params['sparsity_ex'], rng)
        W_in = sparsify(W_in, params['sparsity_in'], rng)
        W_0 = np.concatenate([W_ex, -W_in], 0)
        assert W_0.shape == (params['n_neurons'], params['n_neurons'])
    else:
        raise ValueError()
    if not self_connections:
        np.fill_diagonal(W_0, 0)
    return W_0


def _distance_wrapped(i, j, length):
    """Helper function for mexican_hat"""
    a = (length / 2)
    d = abs(i - j)
    d[d > a] = abs(d[d > a] - length)
    return d

def _mexican_hat(i, j, a, sigma_1, sigma_2, n_neurons):
    """Mexican hat or difference of Gaussians."""
    d = _distance_wrapped(i, j, n_neurons)
    first = np.exp(- d**2 / (2 * sigma_1**2))
    second = a * np.exp(- d**2 / (2 * sigma_2**2))
    return first - second

def construct_mexican_hat_connectivity(params):
    """Construct Mexican hat or difference of Gaussians connectivity matrix.
    Using np.exp(- d**2 / (2 * sigma_1**2)) - a * np.exp(- d**2 / (2 * sigma_2**2))

    Parameters
    ----------
    params : dict
        Must contain the following parameters:
        'mex_a', 'mex_sigma_1', 'mex_sigma_2', 'n_neurons'

    Returns
    -------
    type
        Description of returned object.

    Examples
    --------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    """
    W = np.zeros((params['n_neurons'],params['n_neurons']))
    j = np.arange(params['n_neurons'])
    for i in range(params['n_neurons']):
        W[i,j] = _mexican_hat(
            i, j, *map(params.get,
                ['mex_a', 'mex_sigma_1', 'mex_sigma_2', 'n_neurons']))
    np.fill_diagonal(W, 0)
    return W


def dales_law_transform(W_0):
    """Transform a connectivity matrix such that it follows Dale's law.

    Parameters
    ----------
    W_0 : ndarray
        Connectivity matrix `W_0`.

    Note
    ----
    Doubles the number of neurons by performing the following transform
    W_0 = ((W_0*(W_0>0), W_0*(W_0<0)), (W_0*(W_0>0), W_0*(W_0<0)))

    Returns
    -------
    ndarray
        Transformed connectivity matrix.

    Examples
    --------
    >>> dales_law_transform(np.array([[1,-1], [-1,1]]))
    array([[ 1,  0,  1,  0],
           [ 0,  1,  0,  1],
           [ 0, -1,  0, -1],
           [-1,  0, -1,  0]])
    """
    # Dale's law
    W_0 = np.concatenate((W_0*(W_0>0), W_0*(W_0<0)), 0)
    W_0 = np.concatenate((W_0, W_0), 1)
    return W_0


def sparsify(W_0, sparsity, rng=None, inplace=True):
    """Remove connections making the connectivity matrix more sparse.

    Parameters
    ----------
    W_0 : ndarray
        Connectivity matrix `W_0`.
    sparsity : float
        The percentage of connections to remove should be between 0 and 1.
    rng : generator
        Random number generator if None default is numpy default_rng
        (the default is None).
    inplace : bool
        Make changes to input array or not (the default is True).

    Returns
    -------
    ndarray
        Sparse connectivity.

    Examples
    --------
    >>> rng = np.random.default_rng(1234)
    >>> sparsify(W_0=np.ones((3,3)), sparsity=0.9, rng=rng)
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 1., 0.]])

    """
    rng = default_rng() if rng is None else rng
    indices = np.unravel_index(
        rng.choice(
            np.arange(np.prod(W_0.shape)),
            size=int(sparsity * np.prod(W_0.shape)),
            replace=False
        ),
        W_0.shape
    )
    if inplace:
        W_out = W_0
    else:
        W_out = W_0.copy()
    W_out[indices] = 0
    return W_out


def construct_connectivity_filters(W_0, params):
    """Construct temporal connectivity filters.

    Parameters
    ----------
    W_0 : ndarray
        Base connectivity matrix `W_0`.
    params : dict
        Must contain the following parameters:
        `ref_scale`, `abs_ref_scale`, `abs_ref_strength`, `rel_ref_strength`,
        `spike_scale`, `alpha`.

    Returns
    -------
    ndarray, ndarray, ndarray
        W : The connectivity filter tensor
        excitatory_neuron_idx : excitatory indices
        inhibitory_neuron_idx : inhibitory indices

    Examples
    --------
    >>> params = {'ref_scale': 10, 'abs_ref_scale': 3, 'spike_scale': 5, 'abs_ref_strength': -100, 'rel_ref_strength': -30, 'alpha': 0.2}
    >>> W, eidx, iidx = construct_connectivity_filters(np.array([[0,1],[-1,0]]), params)

    """
    # construct construct connectivity matrix
    W = np.zeros((W_0.shape[0], W_0.shape[1], params['ref_scale']))
    for i in range(W_0.shape[0]):
        for j in range(W_0.shape[1]):
            if i==j:
                W[i, j, :params['abs_ref_scale']] = params['abs_ref_strength']
                abs_ref = np.arange(params['abs_ref_scale'], params['ref_scale'])
                W[i, j, params['abs_ref_scale']:params['ref_scale']] = \
                    params['rel_ref_strength'] * \
                    np.exp(- 0.5 * (abs_ref + params['abs_ref_scale'] + 1))
            else:
                W[i, j, np.arange(params['spike_scale'])] = \
                    W_0[i,j] * \
                    np.exp(-params['alpha']*np.arange(params['spike_scale']))

    excitatory_neuron_idx, = np.where(np.any(W_0 > 0, 1))
    inhibitory_neuron_idx, = np.where(np.any(W_0 < 0, 1))

    return W, excitatory_neuron_idx, inhibitory_neuron_idx


def generate_regular_stim_times(period, size):
    """Generate regular stimulus times.

    Parameters
    ----------
    period : float
        Time between each stimulus onset.
    size : int
        Number of time steps.

    Returns
    -------
    array
        Stimulus times

    Examples
    --------
    It starts after one period
    >>> generate_regular_stim_times(2, 10)
    array([[0., 0., 1., 0., 1., 0., 1., 0., 1., 0.]])
    """
    binned_stim_times = np.zeros(size)
    binned_stim_times[np.arange(period, size, period)] = 1
    binned_stim_times = np.expand_dims(binned_stim_times, 0)
    return binned_stim_times


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


def construct_input_filters(W, strengths, scale):
    """Construct input filters.

    Concatenates along the 0th dimension of the connectivity filter tensor

    Parameters
    ----------
    W : ndarray
        Connectivity filter tensor.
    strengths : dict
        Stimulus strength for each target dict(target_1 = strength_1, ...)
    scale : int
        Temporal scale of input, duration of input filter.

    Returns
    -------
    ndarray
        Connectivity filter tensor with stimulus concatenated at the end of 0th
        dimension.

    Examples
    --------
    >>> W = np.ones((2,2,4))
    >>> W_new = construct_input_filters(W, {0: 2, 1: 2}, 2)
    >>> W_new.shape
    (3, 2, 4)

    """
    W = np.concatenate((W, np.zeros((1, W.shape[1], W.shape[2]))), 0)

    assert isinstance(strengths, dict)

    for j, s in strengths.items():
        W[-1, j, np.arange(scale)] = s
    return W


def simulate(W, W_0, params, inputs=None, pbar=None, rng=None, callback=None):
    """Simulate network activity.

    Parameters
    ----------
    W : ndarray
        Connectivity filter tensor.
    W_0 : ndarray
        Base connectivity matrix.
    params : dict
        Must contain the following parameters
        `ref_scale`, `n_neurons`, `n_time_step`, `const`
    inputs : ndarray
        Stimulus onset times, must correspond to the order of input filters in
        W (the default is None).
    pbar : bool
        Progressbar.
    rng : generator
        Random number generator if None default is numpy default_rng
        (the default is None).

    Returns
    -------
    array
        Spike times and inputs in an event array.

    Examples
    --------
    >>> params = {
    ...     'const': 5.,
    ...     'n_neurons': 3,
    ...     'dt': 1e-3,
    ...     'ref_scale': 10,
    ...     'abs_ref_scale': 3,
    ...     'spike_scale': 5,
    ...     'abs_ref_strength': -100,
    ...     'rel_ref_strength': -30,
    ...     'drive1_targets': [0, 1],
    ...     'drive1_scale': 2,
    ...     'drive1_strength': 10,
    ...     'drive1_period': 50,
    ...     'drive1_isi_min': 10,
    ...     'drive1_isi_max': 200,
    ...     'drive2_targets': [0, 1, 2],
    ...     'drive2_scale': 10,
    ...     'drive2_strength': 5,
    ...     'drive2_period': 100,
    ...     'drive2_isi_min': 30,
    ...     'drive2_isi_max': 400,
    ...     'alpha': 0.2,
    ...     'n_time_step': int(100),
    ...     'seed': 12345
    ... }
    >>> rng = default_rng(params['seed'])
    >>> W_0 = np.array([
    ...     [0, 0, 0],
    ...     [0, 0, 2.],
    ...     [0, 0, 0]
    ... ])
    >>> # set stim
    >>> drive1 = generate_poisson_stim_times(
    ...     params['drive1_period'],
    ...     params['drive1_isi_min'],
    ...     params['drive1_isi_max'],
    ...     params['n_time_step'],
    ...     rng=rng
    ... )
    >>> drive2 = generate_poisson_stim_times(
    ...     params['drive2_period'],
    ...     params['drive2_isi_min'],
    ...     params['drive2_isi_max'],
    ...     params['n_time_step'],
    ...     rng=rng
    ... )
    >>> stimulus = np.concatenate(
    ...     (drive1, drive2), 0)
    >>> W, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)
    >>> W = construct_input_filters(
    ...     W, {i: params['drive1_strength'] for i in params['drive1_targets']},
    ...     params['drive1_scale'])
    >>> W = construct_input_filters(
    ...     W, {i: params['drive2_strength'] for i in params['drive2_targets']},
    ...     params['drive2_scale'])
    >>> # Run the simulation
    >>> simulate(W=W, W_0=W_0, inputs=stimulus, params=params, rng=rng)
    array([[ 0,  9],
           [ 3, 44],
           [ 0, 45],
           [ 1, 45],
           [ 2, 46],
           [ 2, 56],
           [ 1, 66],
           [ 4, 82],
           [ 0, 83],
           [ 1, 84],
           [ 2, 86],
           [ 0, 87],
           [ 1, 89],
           [ 2, 90],
           [ 0, 92]])
    """
    rng = default_rng() if rng is None else rng
    pbar = pbar if pbar is not None else lambda x:x
    x = np.zeros((len(W), params['ref_scale']))
    rand_init = rng.integers(0, 2, params['n_neurons'])
    # if W_0 has dales law transform n_neurons = len(W_0) / 2 and the first
    # half of neurons are excitatory and the second half is their inhibitory
    # copies and thus have to be identically initialized
    if len(W_0) == params['n_neurons'] * 2:
        rand_init = np.concatenate((rand_init, rand_init))

    x[:len(W_0), -1] = rand_init

    if inputs is not None:
        x[len(W_0):] = inputs[:, :x.shape[1]]

    spikes = []
    ref_scale_range = np.arange(params['ref_scale'])

    for t in pbar(range(params['n_time_step'] - 1)):

        if t >= params['ref_scale'] and inputs is not None:
            x[len(W_0):] = inputs[:, t-params['ref_scale']+1: t+1]

        # if any spikes store spike indices and time
        if x[:,-1].any():
            spikes.extend([(idx, t) for idx in np.where(x[:,-1])[0]])

        activation = np.dot(W.T, x)

        activation = activation[ref_scale_range,:,ref_scale_range[::-1]].sum(0)

        #Stimulus has no activation
        activation = activation[:len(W_0)]

        activation = activation - params['const']

        if callback is not None:
            activation = callback(activation)

        x = np.roll(x, -1, 1)

        x[:len(W_0), -1] = rng.binomial(
            1, np.exp(activation) / (np.exp(activation) + 1), size=len(W_0)
        ) # binomial GLM with logit link function (binomial regression)
    return np.array(spikes)


def simulate_torch(W, W_0, params, inputs=None, pbar=None, device='cpu', rng=None):
    """Same as simulate just using PyTorch as the backend
    WARNING: NOT TESTED PROPERLY
    """
    import torch
    rng = torch.Generator() if rng is None else rng
    pbar = pbar if pbar is not None else lambda x:x

    W = torch.as_tensor(W).to(device, dtype=torch.float32)
    if inputs is not None:
        inputs = torch.as_tensor(inputs, dtype=torch.float32).to(device)

    x = torch.zeros((len(W), params['ref_scale']), dtype=torch.float32, device=device)
    rand_init = torch.randint(0, 2, (params['n_neurons'],), generator=rng, device=device)
    # if W_0 has dales law transform n_neurons = len(W_0) / 2 and the first
    # half of neurons are excitatory and the second half is their inhibitory
    # copies and thus have to be identically initialized
    if len(W_0) == params['n_neurons'] * 2:
        rand_init = torch.cat((rand_init, rand_init))

    x[:len(W_0), -1] = rand_init

    if inputs is not None:
        x[len(W_0):] = inputs[:, :x.shape[1]]
    spikes = []


    ref_scale_range = torch.arange(params['ref_scale']).to(device)
    ref_scale_range_flip = ref_scale_range.flip(0)
    for t in pbar(range(params['n_time_step'] - 1)):

        if t >= params['ref_scale'] and inputs is not None:
            x[len(W_0):] = inputs[:, t-params['ref_scale']+1: t+1]

        # if any spikes store spike indices and time
        if x[:,-1].any():
            spikes.extend([(idx.cpu(), t) for idx in torch.where(x[:,-1])[0]])

        activation = torch.einsum('kji,kl->ijl',W,x)

        activation =  activation[ref_scale_range,:,ref_scale_range_flip].sum(0)

        activation = activation - params['const']

        #Stimulus has no activation
        activation = activation[:len(W_0)]

        x = torch.roll(x, -1, 1)

        x[:len(W_0), -1] = torch.bernoulli(
            torch.exp(activation) / (torch.exp(activation) + 1),
            generator=rng
        ) # binomial GLM with logit link function (binomial regression)
    return torch.tensor(spikes).cpu().numpy()
