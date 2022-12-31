from spiking_network.models import SpikingModel
from spiking_network.datasets import W0Dataset, GlorotParams
from spiking_network.utils import tune, new_tune
from spiking_network.stimulation import RegularStimulation

from pathlib import Path
import torch
from torch_geometric.loader import DataLoader

def run_tune(n_neurons, dataset_size, n_steps, n_epochs, model_path, firing_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    rng = torch.Generator().manual_seed(0)
    seeds = {
            "w0": torch.randint(0, 100000, (dataset_size,), generator=rng).tolist(),
            "model": torch.randint(0, 100000, (1,), generator=rng).item(),
         }

    # Parameters for the simulation
    w0_params = GlorotParams(0, 5)
    w0_data = W0Dataset(n_neurons, dataset_size, w0_params, seeds=seeds["w0"])

    # Put the data in a dataloader
    max_parallel = 100
    data_loader = DataLoader(w0_data, batch_size=min(max_parallel, dataset_size), shuffle=False)

    model = SpikingModel(
            seed=seeds["model"],
            device=device
        )
    reg_stim = RegularStimulation(
        targets=torch.randint(0, n_neurons, (10,)).tolist(),
        strengths=torch.rand(10).tolist(),
        intervals = 5,
        temporal_scales=1,
        durations=n_steps,
        total_neurons = min(max_parallel, dataset_size)*n_neurons,
        device=device
    )

    tunable_params = ["alpha", "beta", "threshold", "strengths", "decay"]

    for data in data_loader:
        data = data.to(device)
        model.add_stimulation(reg_stim)
        new_tune(model, data, firing_rate, tunable_parameters=tunable_params)
        #  tune(model, data, firing_rate, tunable_model_parameters=[], stimulation=reg_stim, tunable_stimulation_parameters=["strengths"], n_epochs=n_epochs, n_steps=n_steps, lr=0.1)

    # Save the model
    model.save(model_path / f"{w0_params.name}_{n_neurons}_neurons_{firing_rate}_firing_rate.pt")

