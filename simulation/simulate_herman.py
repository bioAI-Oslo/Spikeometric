import numpy as np
from spiking_network.datasets import HermanDataset
from spiking_network.models import HermanModel

from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from scipy.sparse import coo_matrix

def herman_save(x, model, w0_data, seed, data_path, n_neurons, stimulation=None):
    """Saves the spikes and the connectivity filter to a file"""
    x = torch.cat(x, dim=0)
    x = x.cpu()
    w0_data = w0_data.cpu()
    xs = torch.split(x, n_neurons, dim=0)
    w0s = torch.split(w0_data, n_neurons*n_neurons, dim=0)
    for i, (x, w_0) in enumerate(zip(xs, w0s)):
        sparse_x = coo_matrix(x)
        sparse_W0 = coo_matrix(
            w_0.view(n_neurons, n_neurons),
        )
        np.savez_compressed(
            data_path / Path(f"seed_{seed}_simulation_{i}.npz"),
            X_sparse=sparse_x,
            w_0=sparse_W0,
            stimulation=stimulation.__dict__() if stimulation is not None else None,
            parameters=model.save_parameters(),
            seed=seed,
        )

def mysave(x, model, w0_data, seeds, data_path, stimulation=None):
    """Saves the spikes and the connectivity filter to a file"""
    x = torch.cat(x, dim=0)
    x = x.cpu()
    xs = torch.split(x, w0_data[0].num_nodes, dim=0)
    for i, (x, network) in enumerate(zip(xs, w0_data)):
        sparse_x = coo_matrix(x)
        sparse_W0 = coo_matrix((network.W0, network.edge_index), shape=(network.num_nodes, network.num_nodes))
        np.savez_compressed(
            data_path / Path(f"{i}.npz"),
            X_sparse=sparse_x,
            w_0=sparse_W0,
            stimulation=stimulation.__dict__() if stimulation is not None else None,
            parameters={k: v.item() for k, v in model.state_dict().items()},
            seeds={
                "model": seeds["model"],
                "w0": seeds["w0"][i],
                },
        )


def calculate_isi(spikes: np.ndarray, N, n_steps, dt=0.0001) -> float:
    return N * n_steps * dt / spikes.sum()

def simulate_herman(n_neurons, n_sims, n_steps, data_path, max_parallel, firing_rate=0.16, emptiness=0.9):
    # Path to save results
    data_path = Path(data_path) / Path(f"herman_{n_neurons}_neurons_{n_sims}_sims_{n_steps}_steps")
    data_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    rng = torch.Generator().manual_seed(0)
    seeds = {
        "w0": torch.randint(0, 100000, (n_sims,), generator=rng).tolist(),
        "model": torch.randint(0, 100000, (1,), generator=rng).item(),
     }

    #  model_path = Path("models/saved_models") / f"herman_{n_neurons}_neurons_{firing_rate}_firing_rate.pt"
    #  model = HermanModel(
    #      seed=seeds["model"],
    #      device=device
    #  ).load(model_path) if model_path.exists() else HermanModel(seed=seeds["model"], device=device)
    model = HermanModel(seed=seeds["model"], device=device)

    batch_size = min(n_sims, max_parallel)
    herman_dataset = HermanDataset(n_neurons, n_sims, seed=seeds["w0"], emptiness=emptiness)
    data_loader = DataLoader(herman_dataset, batch_size=batch_size, shuffle=False)
    results = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)

        #  stim = RegularStimulation(targets=0, strengths=1, duration=n_steps, rates=0.2, temporal_scales=2, n_neurons=data.num_nodes, device=device)
        #  spikes = model.simulate(data, n_steps, stimulation=stim)
        spikes = model.simulate(data, n_steps)
        #  print(f"ISI: {calculate_isi(spikes, n_neurons, n_steps)}")
        results.append(spikes)

    #  herman_save(results, model, herman_dataset, seeds, data_path, n_neurons)
    mysave(results, model, herman_dataset, seeds, data_path)
