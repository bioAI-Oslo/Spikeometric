import sys
sys.path.append('..')
from network import SpikingNetwork, FilterParams
import time
import numpy as np

def time_network(network, n_steps, N, total_sims):
    total_time = 0
    for i in range(N):
        s = time.perf_counter()
        network.simulate(n_steps, save_spikes=False, equilibration_steps=0)
        e = time.perf_counter()
        total_time += e - s
    return total_time / total_sims

def main():
    n_steps = 100
    timings = []
    seed = 12345

    filter_params = FilterParams()
    p_sims = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    n_neurons = [10*i for i in range(1, 11)]
    total_sims = p_sims[-1]
    timings = np.zeros((len(p_sims), len(n_neurons)))
    for n in n_neurons:
        for p in p_sims:
            network = SpikingNetwork(n_neurons=n*p, filter_params=filter_params, n_clusters=p, n_cluster_connections = 0, seed=seed)
            time = time_network(network, n_steps, N=total_sims // p, total_sims=total_sims)
            timings[p_sims.index(p), n_neurons.index(n)] = time
            print("p: {}, n: {}, time: {}".format(p, n, time))

    np.savez("benchmark_data" + '/parallelization_compare_across_neurons.npz', timings=timings, p_sims=p_sims)

if __name__ == '__main__':
    main()
