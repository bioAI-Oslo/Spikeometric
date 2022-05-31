import numpy as np
# import torch
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pathlib
import sys
sys.path.append('..')

from generator import (
    construct_connectivity_filters,
    construct_connectivity_matrix,
    construct_input_filters,
    dales_law_transform,
    # simulate_torch,
    simulate
)

def construct(params, rng=None):
    rng = default_rng() if rng is None else rng
    W_0 = construct_connectivity_matrix(params) # Change this to give two populations
    W_0 = dales_law_transform(W_0)
    W, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)

    return W, W_0, excit_idx, inhib_idx


if __name__ == '__main__':
    data_path = pathlib.Path('datasets/')
    data_path.mkdir(parents=True, exist_ok=True)

    params = {
        'const': 5,
        'n_neurons': 10,
        'dt': 1e-3,
        'ref_scale': 10,
        'abs_ref_scale': 3,
        'spike_scale': 5,
        'abs_ref_strength': -100,
        'rel_ref_strength': -30,
        'alpha': 0.2,
        'glorot_normal': {
            'mu': 0,
            'sigma': 5
        },
        'n_time_step': int(1e4),
        'seed': 12345,
    }
    rng = default_rng(params['seed'])

    fname =  f'n10_ss5_s5'

    W, W_0, excit_idx, inhib_idx = construct(params, rng=rng)

    # result = simulate_torch(
    #     W=W,
    #     W_0=W_0,
    #     inputs=stimulus,
    #     params=params,
    #     pbar=True,
    #     device='cuda'
    )
    result = simulate(
        W=W,
        W_0=W_0,
        params=params,
        pbar=True,
    )

    np.savez(
        data_path / fname,
        data=result,
        W=W,
        W_0=W_0,
        params=params,
        excitatory_neuron_idx=excit_idx,
        inhibitory_neuron_idx=inhib_idx
    )
