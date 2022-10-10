import sys
import time
from tqdm import tqdm
import numpy as np
sys.path.append('../')

from network import FilterParams, GlorotParams
from w0_generator import W0Generator
from spiking_model import SpikingModel
from connectivity_filters.connectivity_filter import ConnectivityFilter
import torch

sys.path.append('../..')

from generator import (
    construct_connectivity_filters,
    construct_connectivity_matrix,
    construct_input_filters,
    dales_law_transform,
    simulate
)

def initial_condition(n_neurons, time_scale, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)
    init_cond = torch.randint(0, 2, (n_neurons,), dtype=torch.bool, generator=rng)
    x_initial = torch.zeros((n_neurons, time_scale), dtype=torch.bool)
    x_initial[:, -1] = init_cond
    return x_initial

def construct_mikkel(params, rng=None):
    rng = default_rng() if rng is None else rng
    W_0 = construct_connectivity_matrix(params) # Change this to give two populations
    W_0 = dales_law_transform(W_0)
    W, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)

    return W, W_0, excit_idx, inhib_idx

def time_mikkel(W, W_0, params, rng, N):
    total_time = 0
    for _ in range(N):
        s = time.perf_counter()
        simulate(W, W_0, params=params, rng=rng)
        e = time.perf_counter()
        total_time += e - s
    return total_time / N

def time_network(network, x_initial, n_steps, N=10, parallel=10):
    total_time = 0
    for i in range(N):
        s = time.perf_counter()
        network(x_initial)
        e = time.perf_counter()
        total_time += e - s
    return total_time / (N*parallel)

def main():
    n_steps = 100
    p_sims = 1
    threshold = -5

    # Connectivity parameters
    mu = 0
    sigma = 5
    ref_scale=10
    abs_ref_scale=3
    spike_scale=5
    abs_ref_strength=-100
    rel_ref_strength=-30
    decay_offdiag=0.2
    decay_diag=0.5

    mikkel_params = {
        'const': 5,
        # 'n_neurons': 10,
        'dt': 1e-3,
        'ref_scale': ref_scale,
        'abs_ref_scale': abs_ref_scale,
        'spike_scale': spike_scale,
        'abs_ref_strength': abs_ref_strength,
        'rel_ref_strength': rel_ref_strength,
        'alpha': decay_offdiag,
        'glorot_normal': {
            'mu': mu,
            'sigma': sigma
        },
        'n_time_step': n_steps,
        # 'seed': 12345,
        }

    neuron_list = [10*i for i in range(1, 21)]
    timings = []
    seed = 12345
    for n_neurons in neuron_list:
        # Mikkel
        rng = np.random.default_rng(seed)
        mikkel_params['n_neurons'] = n_neurons // 2
        W_m, W_0, excit_idx, inhib_idx = construct_mikkel(mikkel_params, rng)


        w0_generator = W0Generator(1, n_neurons, 0, GlorotParams(0, 5))
        w0 = w0_generator.generate(n_neurons)
        connectivity_filter = ConnectivityFilter(w0)
        W, edge_index = connectivity_filter.W, connectivity_filter.edge_index

        torch_network = SpikingModel(W, edge_index, n_steps, seed=n_neurons)
        x_initial = initial_condition(n_neurons, connectivity_filter.time_scale, seed)
        mikkel_time = time_mikkel(W=W_m, W_0=W_0, params=mikkel_params, rng=rng, N=1)
        torch_time = time_network(torch_network, x_initial, n_steps, N=1, parallel=1)

        timings.append((mikkel_time, torch_time))
        print(f'{n_neurons} neurons: {mikkel_time} vs {torch_time}')


    np.savez("benchmark_data" + '/new_sim_comp.npz', timings=timings, neuron_list=neuron_list)

if __name__ == '__main__':
    main()
