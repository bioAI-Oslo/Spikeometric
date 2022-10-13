import argparse
from spiking_network.models.spiking_model import SpikingModel
from spiking_network.connectivity_filters.connectivity_filter import ConnectivityFilter
from spiking_network.w0_generators.w0_generator import W0Generator, GlorotParams, NormalParams
from pathlib import Path
from tqdm import tqdm
import torch
from scipy.sparse import coo_matrix
import numpy as np


def initial_condition(n_neurons, time_scale, seed):
    """Initializes the network with a random number of spikes"""
    rng = torch.Generator()
    rng.manual_seed(seed)
    init_cond = torch.randint(0, 2, (n_neurons,), dtype=torch.bool, generator=rng)
    x_initial = torch.zeros((n_neurons, time_scale), dtype=torch.bool)
    x_initial[:, -1] = init_cond
    return x_initial


def save_parallel(
    x, connectivity_filter, n_steps, n_neurons_list, n_edges_list, seed, data_path: str
) -> None:
    """Saves the spikes to a file"""
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    n_clusters = len(n_neurons_list)
    cluster_size = connectivity_filter.n_neurons  // n_clusters

    x_sims = torch.split(x, cluster_size, dim=0)
    Ws = torch.split(connectivity_filter.W, n_edges_list, dim=0)
    edge_indices = torch.split(connectivity_filter.edge_index, n_edges_list, dim=1)
    for i, (x_sim, W_sim, edge_index_sim) in enumerate(zip(x_sims, Ws, edge_indices)):
        sparse_x = coo_matrix(x_sim)
        W0_sim = connectivity_filter.W0[i * cluster_size : (i + 1) * cluster_size, i * cluster_size : (i + 1) * cluster_size]
        sparse_W0 = coo_matrix(W0_sim)
        np.savez_compressed(
            data_path / Path(f"{seed}_{i}.npz"),
            X_sparse=sparse_x,
            W=W_sim,
            edge_index=edge_index_sim - i * cluster_size,
            w_0=sparse_W0,
            seed=seed,
            filter_params=connectivity_filter.parameters,
        )


def save(x, connectivity_filter, n_steps, seed, data_path):
    """Saves the spikes and the connectivity filter to a file"""
    sparse_x = coo_matrix(x)
    sparse_W0 = coo_matrix(connectivity_filter.W0)
    np.savez_compressed(
        data_path,
        X_sparse=sparse_x,
        W=connectivity_filter.W,
        edge_index=connectivity_filter.edge_index,
        w_0=sparse_W0,
        parameters=connectivity_filter.parameters,
        seed=seed,
    )

def make_dataset(n_clusters, cluster_size, n_cluster_connections, n_steps, n_datasets, data_path, is_parallel=False):
    """Generates a dataset"""
    # Set data path
    data_path = (
        Path(data_path)
        / f"n_clusters_{n_clusters}_cluster_size_{cluster_size}_n_cluster_connections_{n_cluster_connections}_n_steps_{n_steps}"
    )
    data_path.mkdir(parents=True, exist_ok=True)

    # Set parameters for W0
    dist_params = GlorotParams(0, 5)
    w0_generator = W0Generator(
        n_clusters, cluster_size, n_cluster_connections, dist_params
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #  device = "cpu"
    with torch.no_grad():  # Disables gradient computation (the models are built on top of torch)
        for i in tqdm(range(n_datasets)):
            w0, n_neurons_list, n_edges_list, hub_neurons = w0_generator.generate(
                i
            )  # Generates a random W0

            connectivity_filter = ConnectivityFilter(
                w0
            )  # Creates a connectivity filter from W0
            W, edge_index = connectivity_filter.W, connectivity_filter.edge_index

            model = SpikingModel(
                W, edge_index, n_steps, seed=i, device=device
            )  # Initializes the model

            x_initial = initial_condition(
                connectivity_filter.n_neurons, connectivity_filter.time_scale, seed=i
            )  # Initializes the network with a random number of spikes
            x_initial = x_initial.to(device)

            spikes = model(x_initial)  # Simulates the network

            if is_parallel:
                save_parallel(spikes, connectivity_filter, n_steps, n_neurons_list, n_edges_list, i, data_path) # Saves the spikes and the connectiv 7ity filter to a file
            else:
                save(spikes, connectivity_filter, n_steps, i, data_path / Path(f"{i}.npz")) # Saves the spikes and the connectivity filter to a file

if __name__ == "__main__":
    make_dataset(10, 20, 1, 1000, 1)
