from spiking_network.models import GLMModel
from spiking_network.datasets import NormalConnectivityDataset, GlorotParams
from spiking_network.stimulation import RegularStimulation
from spiking_network.utils import simulate, save_data

import torch_geometric.transforms as T
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader

def run_simulation(n_neurons, n_sims, n_steps, data_path, folder_name, seed, max_parallel=100):
    """Generates a dataset"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    rng = torch.Generator().manual_seed(seed)
    seeds = {
            "w0": torch.randint(0, 100000, (1,), generator=rng).item(),
            "model": torch.randint(0, 100000, (1,), generator=rng).item(),
         }

    # Set path for saving data
    data_path = Path(data_path) / (folder_name if folder_name else Path(f"{n_neurons}_neurons_{n_sims}_datasets_{n_steps}_steps"))
    data_path.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    w0_params = GlorotParams(0, 5)
    data_root = f"data/w0/{n_neurons}_{n_sims}_{w0_params.name}_{w0_params.mean}_{w0_params.std}_{seeds['w0']}"
    w0_data = NormalConnectivityDataset(n_neurons, n_sims, w0_params, seed=seeds["w0"], root=data_root)
    data_loader = DataLoader(w0_data, batch_size=min(n_sims, max_parallel), shuffle=False)

    model = GLMModel(seed=seeds["model"], device=device)

    results = []
    for data in data_loader:
        data = data.to(device)
        spikes = simulate(model, data, n_steps)
        print(f"Firing rate: {spikes.sum() / (n_steps * data.num_nodes):.5f}")
        results.append(spikes)

    save_data(results, model, w0_data, seeds, data_path)
