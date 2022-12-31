import numpy as np
from spiking_network.datasets import HermanDataset
from spiking_network.models import HermanModel
from spiking_network.utils import simulate, save_data, calculate_firing_rate

from pathlib import Path
import torch
from torch_geometric.loader import DataLoader

def run_herman(n_neurons, n_sims, n_steps, data_path, folder_name, max_parallel, emptiness=0.9):
    # Path to save results
    data_path = Path(data_path) / (folder_name if folder_name else Path(f"herman_{n_neurons}_neurons_{n_sims}_sims_{n_steps}_steps"))
    data_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    rng = torch.Generator().manual_seed(0)
    seeds = {
        "w0": torch.randint(0, 100000, (1,), generator=rng).item(),
        "model": torch.randint(0, 100000, (1,), generator=rng).item(),
     }

    model = HermanModel(seed=seeds["model"], device=device)

    batch_size = min(n_sims, max_parallel)
    herman_dataset = HermanDataset(n_neurons, n_sims, seed=seeds["w0"], emptiness=emptiness)
    data_loader = DataLoader(herman_dataset, batch_size=batch_size, shuffle=False)
    results = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)

        spikes = simulate(model, data, n_steps)
        print("Firing rate", calculate_firing_rate(spikes))
        results.append(spikes)

    save_data(results, model, herman_dataset, seeds, data_path)
