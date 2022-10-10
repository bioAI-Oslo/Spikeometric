import argparse
from spiking_network.models.spiking_model import SpikingModel
from spiking_network.connectivity_filters.connectivity_filter import ConnectivityFilter
from spiking_network.w0_generators.w0_generator import W0Generator, GlorotParams
from pathlib import Path
from tqdm import tqdm
import torch
from scipy.sparse import coo_matrix
import numpy as np

def initial_condition(n_neurons, time_scale, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)
    init_cond = torch.randint(0, 2, (n_neurons,), dtype=torch.bool, generator=rng)
    x_initial = torch.zeros((n_neurons, time_scale), dtype=torch.bool)
    x_initial[:, -1] = init_cond
    return x_initial

def save(spikes, connectivity_filter, n_steps, seed, data_path:str) -> None:
    x = spikes[0]
    t = spikes[1]
    data = torch.ones_like(t)
    sparse_x = coo_matrix((data, (x, t)), shape=(connectivity_filter.W0.shape[0], n_steps))
    np.savez_compressed(
            data_path,
            X_sparse = sparse_x,
            W=connectivity_filter.W,
            edge_index=connectivity_filter.edge_index,
            parameters = connectivity_filter.parameters,
            seed=seed,
        )

def make_dataset(n_clusters, cluster_size, n_cluster_connections, n_steps, n_datasets, data_path):
    data_path = Path(data_path) / f"n_clusters_{n_clusters}_cluster_size_{cluster_size}_n_cluster_connections_{n_cluster_connections}_n_steps_{n_steps}"
    data_path.mkdir(parents=True, exist_ok=True)
    
    dist_params = GlorotParams(0, 5)
    w0_generator = W0Generator(n_clusters, cluster_size, n_cluster_connections, dist_params)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for i in tqdm(range(n_datasets)):
            w0 = w0_generator.generate(i)
            connectivity_filter = ConnectivityFilter(w0)
            W, edge_index = connectivity_filter.W, connectivity_filter.edge_index

            model = SpikingModel(W, edge_index, n_steps, seed=i, device=device)

            x_initial = initial_condition(connectivity_filter.n_neurons, connectivity_filter.time_scale, seed=i)
            x_initial = x_initial.to(device)

            spikes = model(x_initial)
            save(spikes, connectivity_filter, n_steps, i, data_path / Path(f"{i}.npz"))

if __name__ == "__main__":
    make_dataset(10, 20, 1, 1000, 1)
