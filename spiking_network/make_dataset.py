import argparse
from spiking_network.network import SpikingNetwork, GlorotParams
from spiking_network.network.rolling_network import RollingNetwork
from spiking_network.connectivity_filters import ConnectivityFilter
from spiking_network.w0_generator import W0Generator
from pathlib import Path
from tqdm import tqdm
from spiking_network.save_data import save, save_parallel
import torch

def make_dataset(n_clusters, cluster_size, n_cluster_connections, n_steps, n_datasets):
    # Path to save results
    dataset_path = (
        Path("/scratch/users/hermabr/data")
        if Path("/scratch/users/hermabr").exists()
        else Path("spiking_network/data")
    )

    data_path = dataset_path / Path(f"jakob_{n_clusters*cluster_size}_neurons_{n_steps}_steps")
    data_path.mkdir(parents=True, exist_ok=True)
    
    dist_params = GlorotParams(0, 5)
    filter_params = {"alpha": 0.2, "beta": 0.5, "abs_ref_strength": -100.0, "rel_ref_strength": -30.0}
    
    for i in tqdm(range(n_datasets)):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        w0, n_neurons_list, n_edges_list, hub_neurons = W0Generator.generate(n_clusters, cluster_size, n_cluster_connections, dist_params, seed=i)
        connectivity_filter = ConnectivityFilter(w0, filter_params)
        network = SpikingNetwork(connectivity_filter, seed=i, device=device)

        network.simulate(n_steps=n_steps)
        network.save(data_path)

if __name__ == "__main__":
    make_dataset(10, 20, 1, 1000)
