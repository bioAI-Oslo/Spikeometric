import sys
import time
sys.path.append('..')
from tqdm import tqdm
import numpy as np

from simulator import *
from models import *
from connectivity import *

sys.path.append('../..')

from generator import (
    construct_connectivity_filters,
    construct_connectivity_matrix,
    construct_input_filters,
    dales_law_transform,
    # simulate_torch,
    simulate
)

def construct_mikkel(params, rng=None):
    rng = default_rng() if rng is None else rng
    W_0 = construct_connectivity_matrix(params) # Change this to give two populations
    W_0 = dales_law_transform(W_0)
    W, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)

    return W, W_0, excit_idx, inhib_idx

def construct_filter(connectivity_filter_generator, p_sims, rng):
    W, edge_index = connectivity_filter_generator.new_filter(p_sims, rng)
    return W, edge_index

def time_mikkel(W, W_0, params, rng, N):
    total_time = 0
    for _ in range(N):
        s = time.time()
        simulate(W, W_0, params=params, rng=rng)
        e = time.time()
        total_time += e - s
    return total_time / N

def time_graph(W, edge_index, simulator, N):
    total_time = 0
    s = time.time()
    simulator.run(W, edge_index, N)
    e = time.time()
    total_time += e - s
    return total_time / N

def main():
    n_steps = 1000
    threshold = -5
    n_neurons = 20

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

    normal_params = NormalParams(mu, sigma)
    filter_params = FilterParams(
            ref_scale=ref_scale,
            abs_ref_scale=abs_ref_scale,
            spike_scale=spike_scale,
            abs_ref_strength=abs_ref_strength,
            rel_ref_strength=rel_ref_strength,
            decay_offdiag=decay_offdiag,
            decay_diag=decay_diag
        )

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
        'n_time_step': n_steps
        # 'seed': 12345,
        }


    sims_list = [10*i for i in range(1, 11)]
    timings = []

    rng_t = torch.Generator().manual_seed(123)
    rng_np = np.random.default_rng(123)
    for n_sims in sims_list:
        # Mikkel
        mikkel_params['n_neurons'] = n_neurons // 2
        W_m, W_0, excit_idx, inhib_idx = construct_mikkel(mikkel_params, rng_np)

        # Torch
        cf_generator = ConnectivityFilterGenerator(n_neurons, normal_params, filter_params)
        W_g, edge_index = construct_filter(cf_generator, n_sims, rng_t)
        simulator_torch = TorchSimulator(n_steps, n_sims, n_neurons, threshold)

        # Numpy
        W_g_numpy, edge_index_numpy = W_g.numpy(), edge_index.numpy()
        simulator_numpy = NumpySimulator(n_steps, n_sims, n_neurons, threshold)

        mikkel_time = time_mikkel(W=W_m, W_0=W_0, params=mikkel_params, rng=rng_np, N=n_sims)
        torch_time = time_graph(W=W_g, edge_index=edge_index, simulator = simulator_torch, N=n_sims)
        numpy_time = time_graph(W=W_g_numpy, edge_index=edge_index_numpy, simulator = simulator_numpy, N=n_sims)
        timings.append([mikkel_time, torch_time, numpy_time])
        print(f"n_neurons: {n_neurons}, Mikkel: {mikkel_time:.3f}, Torch: {torch_time:.3f}, Numpy: {numpy_time:.3f}")


    np.savez('time_data/' + 'sims_dependence.npz', timings=timings, sims_list=sims_list)

if __name__ == '__main__':
    main()
