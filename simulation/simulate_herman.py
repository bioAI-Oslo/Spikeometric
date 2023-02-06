from spiking_network.datasets import UniformGenerator
from spiking_network.models import SAModel
from spiking_network.utils import save_data, calculate_firing_rate, calculate_isi
from config_params import lnp_params

from pathlib import Path
import torch
from torch_geometric.loader import DataLoader


def run_herman(n_neurons, n_sims, n_steps, data_path, folder_name, seed, max_parallel=100, sparsity=0.9):
    # Path to save results
    data_path = Path(data_path) / (folder_name if folder_name else Path(f"lnp_{n_neurons}_neurons_{n_sims}_sims_{n_steps}_steps"))
    data_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    rng = torch.Generator().manual_seed(seed)
    model = SAModel(lnp_params, rng=rng)

    batch_size = min(n_sims, max_parallel)
    data_path = f"data/w0/{n_neurons}_{n_sims}_uniform_{sparsity}_{seed}"
    low = -0.002289225919299652
    high = 0
    uniform_dataset = UniformGenerator(n_neurons, n_sims, low, high, sparsity=sparsity, rng=rng, root=data_path)
    data_loader = DataLoader(uniform_dataset, batch_size=batch_size, shuffle=False)
    results = []
    for data in data_loader:
        data = data.to(device)
        model.to(device)
        spikes = model.simulate(data, n_steps=n_steps)
        print("Firing rate", calculate_firing_rate(spikes))
        results.append(spikes)

    save_data(results, model, uniform_dataset, seed, data_path)
