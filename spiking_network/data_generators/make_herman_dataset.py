import numpy as np
from spiking_network.w0_generators.w0_generator import W0Generator
from spiking_network.models.herman_model import HermanModel
from spiking_network.connectivity_filters.herman_filter import HermanFilter
from pathlib import Path
from tqdm import tqdm
import torch
from scipy.sparse import coo_matrix

def sparse_weight_matrix(N: int):
    mexican_hat_lowest = -0.002289225919299652
    mat = np.random.uniform(mexican_hat_lowest, 0, size=(n, n))
    mat[np.random.rand(*mat.shape) < 0.9] = 0
    return mat

def save(spikes, connectivity_filter, n_steps, seed, data_path):
    """Saves the spikes and the connectivity filter to a file"""
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

def save_parallel(x, connectivity_filter, n_steps, n_neurons_list, n_edges_list, seed, data_path: Path) -> None:
    """Saves the spikes to a file"""
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    n_clusters = len(n_neurons_list)
    cluster_size = connectivity_filter.n_neurons  // n_clusters

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
                filter_params = connectivity_filter.parameters
)

def calculate_isi(spikes: np.ndarray, N, n_steps, dt=0.0001) -> float:
    return N * n_steps * dt / spikes.sum()

def make_herman_dataset(n_sims, N, r, threshold, n_steps, n_datasets, data_path, is_parallel=False):
    # Path to save results
    data_path = Path(data_path) / f"herman_{N}_{r}_{n_steps}"
    data_path.mkdir(parents=True, exist_ok=True)

    #  device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    with torch.no_grad():
        for i in tqdm(range(n_datasets), leave=False):
            w0, n_neurons_list, n_edges_list = W0Generator.generate_herman(n_sims, N, i)

            noise_sparsity = 1.0

            connectivity_filter = HermanFilter(w0,N=N,nsteps=n_steps, noise_sparsity=noise_sparsity)
            W, edge_index = connectivity_filter.W, connectivity_filter.edge_index

            model = HermanModel(
                    W,
                    edge_index,
                    r=r,
                    threshold=threshold,
                    n_steps=n_steps,
                    seed=i,
                    device=device,
                    noise_sparsity=noise_sparsity
                )

            act_initial = torch.zeros((N*n_sims,1), dtype=torch.float32, device=device)
            act_initial = act_initial.to(device)

            spikes = model(act_initial)

            if is_parallel:
                save_parallel(spikes, connectivity_filter, n_steps, n_neurons_list, n_edges_list, i, data_path)
            else:
                print("isi:",calculate_isi(spikes, N, n_steps))
                save(spikes, connectivity_filter, n_steps, i, data_path / Path(f"{i}.npz"))
