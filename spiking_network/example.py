from network import SpikingNetwork, FilterParams
import torch
from tqdm import tqdm

def main():
    filter_params = FilterParams()
    n_clusters = 1
    n_neurons = 500
    n_cluster_connections = 0
    for i in tqdm(range(10)):
        network = SpikingNetwork(n_neurons, filter_params, n_clusters, n_cluster_connections, seed=i)
        network.simulate(n_steps=1000, data_path="data/example_data/", is_parallel=False)

if __name__ == '__main__':
    main()
