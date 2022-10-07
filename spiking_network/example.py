from network import SpikingNetwork, GlorotParams
import torch
from connectivity_filters.connectivity_filter import ConnectivityFilter
from w0_generator import W0Generator
from tqdm import tqdm

def main():
    dist_params = GlorotParams(0, 5)
    n_clusters = 6
    cluster_size = 20
    n_cluster_connections = 1
    filter_params = {"alpha": 0.5, "beta": 0.5, "abs_ref_strength": -100, "rel_ref_strength": -30, "threshold": -5}
    for i in tqdm(range(10)):
        w0 = W0Generator.generate(n_clusters, cluster_size, n_cluster_connections, dist_params, seed=i)
        connectivity_filter = ConnectivityFilter(w0, 10, filter_params)
        network = SpikingNetwork(connectivity_filter, seed=i)
        network.simulate(n_steps=10000, data_path="data/example_data/")

if __name__ == '__main__':
    main()
