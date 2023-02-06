from spiking_network.models import GLMModel
from spiking_network.datasets import NormalConnectivityDataset
from spiking_network.utils import tune
from spiking_network.stimulation import RegularStimulation, SinStimulation, PoissonStimulation
from config_params import glm_params

from pathlib import Path
import torch
from torch_geometric.loader import DataLoader

def run_tune(n_neurons, dataset_size, n_steps, n_epochs, model_path, firing_rate, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    rng = torch.Generator().manual_seed(seed)

    # Parameters for the simulation
    mu = 0
    sigma = 5
    data_path = f"data/w0/{n_neurons}_{dataset_size}_glorot_{mu}_{sigma}_{seed}"
    w0_data = NormalConnectivityDataset(n_neurons, dataset_size, mu=mu, sigma=sigma, glorot=True, rng=rng, root=data_path)

    # Put the data in a dataloader
    max_parallel = 100
    data_loader = DataLoader(w0_data, batch_size=min(max_parallel, dataset_size), shuffle=False)
    
    model = GLMModel(glm_params, rng=rng)
    
    tunable_params = ["alpha", "beta", "threshold"]

    for data in data_loader:
        data = data.to(device)
        model.to(device)
        model.tune(data, firing_rate, tunable_parameters=tunable_params, n_epochs=n_epochs, n_steps=100, lr=0.1)

    # Save the model
    model.save(model_path / f"glorot_{n_neurons}_neurons_{firing_rate}_firing_rate.pt")

