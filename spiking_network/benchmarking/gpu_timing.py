import sys
import time
from tqdm import tqdm
import numpy as np

sys.path.append('..')
from spiking_network.network import SpikingNetwork, GlorotParams
from spiking_network.connectivity_filters import ConnectivityFilter
from spiking_network.w0_generator import W0Generator


def time_network(network, n_steps, N=10, parallel=10, device='cpu'):
    total_time = 0
    for i in range(N):
        s = time.perf_counter()
        network.simulate(n_steps, equilibration_steps=0, device=device)
        e = time.perf_counter()
        total_time += e - s
    return total_time / N

def main():
    n_steps = 100
    dist_params = GlorotParams(0, 5)
    filter_params = {"alpha": 0.2, "beta": 0.5, "threshold": -5, "abs_ref_strength": -100, "rel_ref_strength": -30}

    # Connectivity parameters
    neuron_list = [1000*i for i in range(1, 21)]
    timings = []
    for n_neurons in neuron_list:
        w0, _, _ = W0Generator.generate(1, n_neurons, 0, dist_params)
        connectivity_filter = ConnectivityFilter(w0, 10, filter_params)

        network = SpikingNetwork(connectivity_filter, seed = n_neurons)

        time = time_network(network, n_steps, N=1, device='cuda')

        timings.append(time)
        print(f'{n_neurons} neurons: {time}')

    np.savez("benchmarking/benchmark_data" + '/gpu_timing.npz', timings=timings, neuron_list=neuron_list)

if __name__ == '__main__':
    main()
