from pathlib import Path
from scipy.sparse import coo_matrix
import numpy as np
import torch


def save_parallel(
    x, connectivity_filter, n_steps, n_neurons_list, n_edges_list, seed, data_path: str
) -> None:
    """Saves the spikes to a file"""
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    n_clusters = len(n_neurons_list)
    cluster_size = connectivity_filter.n_neurons // n_clusters

    x_sims = torch.split(x, cluster_size, dim=0)
    Ws = torch.split(connectivity_filter.W, n_edges_list, dim=0)
    edge_indices = torch.split(connectivity_filter.edge_index, n_edges_list, dim=1)
    for i, (x_sim, W_sim, edge_index_sim) in enumerate(zip(x_sims, Ws, edge_indices)):
        sparse_x = coo_matrix(x_sim)
        W0_sim = connectivity_filter.W0[
            i * cluster_size : (i + 1) * cluster_size,
            i * cluster_size : (i + 1) * cluster_size,
        ]
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


def save(x, model, w0_data, seed, data_path, stimulation=None):
    """Saves the spikes and the connectivity filter to a file"""
    x = x.cpu()
    xs = torch.split(x, w0_data[0].num_nodes, dim=0)
    for i, (x, network) in enumerate(zip(xs, w0_data)):
        sparse_x = coo_matrix(x)
        sparse_W0 = coo_matrix(
            (network.W0, network.edge_index),
            shape=(network.num_nodes, network.num_nodes),
        )
        np.savez_compressed(
            data_path / Path(f"seed_{seed}_simulation_{i}.npz"),
            X_sparse=sparse_x,
            w_0=sparse_W0,
            stimulation=stimulation.__dict__() if stimulation is not None else None,
            parameters=model.save_parameters(),
            seed=seed,
        )
