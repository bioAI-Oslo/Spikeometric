from spiking_network.models import BernoulliGLM
from spiking_network.datasets import NormalGenerator
from spiking_network.stimulation import RegularStimulation
from spiking_network.utils import save_data, ConnectivityLoader

from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from config_params import glm_params


def run_simulation(n_neurons, n_sims, n_steps, data_path, folder_name, seed, max_parallel=100):
    """Generates a dataset"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    rng = torch.Generator().manual_seed(seed)
    # Set path for saving data
    data_path = Path(data_path) / (folder_name if folder_name else Path(f"{n_neurons}_neurons_{n_sims}_datasets_{n_steps}_steps"))
    data_path.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    mu = 0
    sigma = 5
    data_root = f"data/w0/{n_neurons}_{n_sims}_glorot_{mu}_{sigma}_{seed}"
    w0_data = NormalGenerator(n_neurons, n_sims, mu=mu, sigma=sigma, glorot=True, rng=rng, root=data_root)
    data_loader = DataLoader(w0_data, batch_size=min(500, n_sims), shuffle=False)

    model = BernoulliGLM(parameters=glm_params, rng=rng)
    results = []
    for data in data_loader:
        data = data.to(device)
        model.to(device)
        spikes = model.simulate(data, n_steps)
        print(f"Firing rate: {spikes.sum() / (n_steps * data.num_nodes):.5f}")
        results.append(spikes)

    save_data(results, model, w0_data, seed, data_path)
