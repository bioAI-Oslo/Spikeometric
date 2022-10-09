from pathlib import Path
import torch
from scipy.sparse import coo_matrix
import numpy as np

def save(spikes, connectivity_filter, n_steps, seed, data_path:str) -> None:
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    x = spikes[0]
    t = spikes[1]
    data = torch.ones_like(t)
    sparse_x = coo_matrix((data, (x, t)), shape=(connectivity_filter.W0.shape[0], n_steps))
    np.savez(
            data_path / Path(f"{seed}.npz"),
            X_sparse = sparse_x,
            W=connectivity_filter.W,
            edge_index=connectivity_filter.edge_index,
            filter_params = connectivity_filter.filter_parameters,
            seed=seed,
        )


def save_parallel(x, connectivity_filter, n_neurons_list, n_edges_list, seed, data_path: str) -> None:
    """Saves the spikes to a file"""
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    n_clusters = len(n_neurons_list)

    n_neurons = x.shape[0] // n_clusters
    x = x[:, connectivity_filter.time_scale:]
    x_sims = torch.split(x, n_neurons_list, dim=0)
    Ws = torch.split(connectivity_filter.W, n_edges_list, dim=0)
    edge_indices = torch.split(connectivity_filter.edge_index, n_edges_list, dim=1)
    for i, (x_sim, W_sim, edge_index_sim) in enumerate(zip(x_sims, Ws, edge_indices)):
        sparse_x = coo_matrix(x_sim)
        np.savez(
                data_path / Path(f"{seed}_{i}.npz"),
                X_sparse = sparse_x,
                W=W_sim,
                edge_index=edge_index_sim,
                seed=seed,
                filter_params = connectivity_filter.filter_parameters
            )
