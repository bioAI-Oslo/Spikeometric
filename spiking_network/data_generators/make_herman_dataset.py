import numpy as np
from spiking_network.models.herman_model import HermanModel
from spiking_network.connectivity_filters.herman_filter import HermanFilter
from pathlib import Path
from tqdm import tqdm
import torch
from scipy.sparse import coo_matrix

def sparse_weight_matrix(N: int):
    MEXICAN_HAT_LOWEST = -0.002289225919299652
    mat = np.random.uniform(MEXICAN_HAT_LOWEST, 0, size=(N, N))
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

def make_herman_dataset(N, r, n_steps, n_datasets, data_path):
    # Path to save results
    data_path = Path(data_path) / f"herman_{N}_{r}_{n_steps}"
    data_path.mkdir(parents=True, exist_ok=True)

    #  device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    with torch.no_grad():
        for i in tqdm(range(n_datasets)):
            w0 = sparse_weight_matrix(N)
            w0 = torch.tensor(w0, dtype=torch.float32)

            connectivity_filter = HermanFilter(w0)
            W, edge_index = connectivity_filter.W, connectivity_filter.edge_index

            model = HermanModel(
                    W,
                    edge_index,
                    r=r,
                    n_steps=n_steps,
                    seed=i,
                    device=device
                )

            act_initial = torch.zeros((N,1), dtype=torch.float32, device=device)
            act_initial = act_initial.to(device)

            spikes = model(act_initial)
            from IPython import embed; embed()
            save(spikes, connectivity_filter, n_steps, i, data_path / Path(f"{i}.npz"))

if __name__ == "__main__":
    make_dataset(10, 20, 1, 1000, 1)
