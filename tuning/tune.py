from spiking_network.models import GLMModel
from spiking_network.datasets import NormalConnectivityDataset, GlorotParams
from spiking_network.utils import tune

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
            "w0": torch.randint(0, 100000, (dataset_size,), generator=rng).tolist(),
            "model": torch.randint(0, 100000, (1,), generator=rng).item(),
         }

    # Parameters for the simulation
    w0_params = GlorotParams(0, 5)
    w0_data = NormalConnectivityDataset(n_neurons, dataset_size, w0_params, seeds=seeds["w0"])

    # Put the data in a dataloader
    max_parallel = 100
    data_loader = DataLoader(w0_data, batch_size=min(max_parallel, dataset_size), shuffle=False)

    tuneable_params = ["alpha", "beta", "threshold"]
    model = GLMModel(
            tuneable_parameters=tuneable_params,
            seed=seeds["model"],
            device=device
        )

    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        tune(model, data, firing_rate, n_epochs=n_epochs, n_steps=n_steps, lr=0.1)

    # Save the model
    model.save(model_path / f"{w0_params.name}_{n_neurons}_neurons_{firing_rate}_firing_rate.pt")

