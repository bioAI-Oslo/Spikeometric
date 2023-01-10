from spiking_network.models import GLMModel
from spiking_network.datasets import NormalConnectivityDataset, GlorotParams
from spiking_network.utils import tune
from spiking_network.stimulation import RegularStimulation, SinStimulation, PoissonStimulation

from pathlib import Path
import torch
from torch_geometric.loader import DataLoader

def run_tune(n_neurons, dataset_size, n_steps, n_epochs, model_path, firing_rate, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    rng = torch.Generator().manual_seed(seed)
    seeds = {
            "w0": torch.randint(0, 100000, (1,), generator=rng).item(),
            "model": torch.randint(0, 100000, (1,), generator=rng).item(),
         }

    # Parameters for the simulation
    w0_params = GlorotParams(0, 5)
    data_path = f"data/w0/{n_neurons}_{dataset_size}_{w0_params.name}_{w0_params.mean}_{w0_params.std}_{seeds['w0']}"
    w0_data = NormalConnectivityDataset(n_neurons, dataset_size, w0_params, seed=seeds["w0"], root=data_path)

    # Put the data in a dataloader
    max_parallel = 100
    data_loader = DataLoader(w0_data, batch_size=min(max_parallel, dataset_size), shuffle=False)
    
    model = GLMModel(
            seed=seeds["model"],
            device=device,
    )
    
    tunable_params = ["alpha", "beta", "threshold"]

    for data in data_loader:
        data = data.to(device)
        tune(model, data, firing_rate, tunable_parameters=tunable_params, n_steps=n_steps, n_epochs=n_epochs, lr=0.1)

    # Save the model
    model.save(model_path / f"{w0_params.name}_{n_neurons}_neurons_{firing_rate}_firing_rate.pt")

