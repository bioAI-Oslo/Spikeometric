import numpy as np
from spiking_network.models.herman_model import HermanModel
from spiking_network.connectivity_filters.herman_filter import HermanFilter
from pathlib import Path
from tqdm import tqdm
import torch

def sparse_weight_matrix(N: int):
    MEXICAN_HAT_LOWEST = -0.002289225919299652
    mat = np.random.uniform(MEXICAN_HAT_LOWEST, 0, size=(N, N))
    mat[np.random.rand(*mat.shape) < 0.9] = 0
    return mat

def make_herman_dataset(N, r, n_steps, n_datasets, data_path):
    # Path to save results
    data_path = Path(data_path) / f"herman_{N}_{r}_{n_steps}"
    data_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for i in tqdm(range(n_datasets)):
            w0 = sparse_weight_matrix(N)
            w0 = torch.tensor(w0, dtype=torch.float32, device=device)

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

            x_initial = torch.zeros((N, 1), dtype=torch.bool, device=device)
            x_initial = x_initial.to(device)

            spikes = model(x_initial)
            save(spikes, connectivity_filter, n_steps, i, data_path / Path(f"{i}.npz"))

if __name__ == "__main__":
    make_dataset(10, 20, 1, 1000, 1)
