import argparse
from spiking_network.network import SpikingNetwork, GlorotParams
from spiking_network.connectivity_filters import ConnectivityFilter
from spiking_network.w0_generator import W0Generator
from pathlib import Path
from tqdm import tqdm
from spiking_network.save_data import save, save_parallel
import torch

def make_dataset(n_clusters, cluster_size, n_cluster_connections, n_steps, n_datasets, is_parallel=False):
    # Path to save results
    dataset_path = (
        Path("/scratch/users/hermabr/data")
        if Path("/scratch/users/hermabr").exists()
        else Path("data")
    )

    data_path = dataset_path / Path(f"jakob_{n_clusters*cluster_size}_neurons_{n_steps}_steps")
    data_path.mkdir(parents=True, exist_ok=True)
    
    dist_params = GlorotParams(0, 5)
    filter_params = {"alpha": 0.2, "beta": 0.5, "abs_ref_strength": -100, "rel_ref_strength": -30, "threshold": -5}
    
    if is_parallel:
        n_clusters = 50
        if n_datasets % n_clusters == 0:
            n_datasets = n_datasets // n_clusters
        else:
            raise ValueError("n_datasets must be divisible by n_clusters to run in parallel")

    for i in tqdm(range(n_datasets)):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        w0, n_neurons_list, n_edges_list = W0Generator.generate(n_clusters, cluster_size, n_cluster_connections, dist_params, seed=i)
        connectivity_filter = ConnectivityFilter(w0, 10, filter_params)
        network = SpikingNetwork(connectivity_filter, seed=i)
        network.simulate(n_steps=n_steps, data_path=data_path, device=device)
        if is_parallel:
            save_parallel(network.x, connectivity_filter, n_neurons_list, n_edges_list, i, data_path)
        else:
            save(network.x, connectivity_filter, i, data_path)

if __name__ == "__main__":
    make_dataset(10, 20, 1, 1000)
