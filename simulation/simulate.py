from spiking_network.models import GLMModel
from spiking_network.datasets import NormalConnectivityDataset, GlorotParams
from spiking_network.stimulation import RegularStimulation, PoissonStimulation, MixedStimulation
from spiking_network.utils import simulate, save

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
            "w0": torch.randint(0, 100000, (n_sims,), generator=rng).tolist(),
            "model": torch.randint(0, 100000, (1,), generator=rng).item(),
         }

    # Set path for saving data
    data_path = Path(data_path) / (folder_name if folder_name else Path(f"{n_neurons}_neurons_{n_sims}_datasets_{n_steps}_steps"))
    data_path.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    w0_params = GlorotParams(0, 5)
    w0_data = NormalConnectivityDataset(n_neurons, n_sims, w0_params, seeds=seeds["w0"])
    data_loader = DataLoader(w0_data, batch_size=min(n_sims, max_parallel), shuffle=False)

    #  model_path = Path("data/saved_models") / f"{w0_params.name}_{n_neurons}_neurons_{firing_rate}_firing_rate.pt"
    model = GLMModel(seed=seeds["model"], device=device)

    results = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        spikes = simulate(model, data, n_steps, stimulation=None)
        print(f"Firing rate: {spikes.sum() / (n_steps * data.num_nodes):.5f}")
        results.append(spikes)

    save(results, model, w0_data, seeds, data_path)
