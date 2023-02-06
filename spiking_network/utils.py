import torch
import torch.nn as nn
from tqdm import tqdm
from spiking_network.models import BaseModel
import numpy as np
from torch_geometric.loader import DataLoader
from scipy.sparse import coo_matrix
from pathlib import Path

def load_data(file):
    """
    Loads the data from the given file.

    Parameters:
    ----------
    file: str

    Returns:
    -------
    X: torch.Tensor
    W0: torch.Tensor
    """
    data = np.load(file, allow_pickle=True)

    X_sparse = data["X_sparse"].item()
    X = X_sparse.toarray()

    W0_sparse = data["w_0"].item()
    W0 = W0_sparse.toarray()

    return torch.from_numpy(X), torch.from_numpy(W0)

def save_data(x, model, w0_data, seed, data_path, stimulation=None):
    """Saves the spikes and the connectivity filter to a file"""
    if not isinstance(x, torch.Tensor):
        x = torch.cat(x, dim=0)
    x = x.cpu()
    xs = torch.split(x, w0_data[0].num_nodes, dim=0)
    for i, (x, network) in enumerate(zip(xs, w0_data)):
        sparse_x = coo_matrix(x)
        sparse_W0 = coo_matrix((network.W0, network.edge_index), shape=(network.num_nodes, network.num_nodes))
        np.savez_compressed(
            data_path / Path(f"{i}.npz"),
            X_sparse=sparse_x,
            w_0=sparse_W0,
            parameters=dict(model.state_dict()),
            stimulation=stimulation.parameter_dict if stimulation else None,
            seed=seed
        )

def calculate_isi(spikes, dt=0.001) -> float:
    """
    Calculates the interspike interval of the network.

    Parameters:
    ----------
    spikes: torch.Tensor
        The spikes of the network
    N: int
        The number of neurons in the network
    n_steps: int
        The number of time steps the network was simulated for
    dt: float
        The time step size
    """
    N = spikes.shape[0]
    n_steps = spikes.shape[1]
    return N * n_steps * dt / spikes.sum()

def calculate_firing_rate(spikes, dt) -> float:
    return (spikes.float().mean() / dt) * 1000