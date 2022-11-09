from spiking_network.models import SpikingModel
from spiking_network.datasets import W0Dataset, GlorotParams, SparseW0Dataset
from spiking_network.stimulation import RegularStimulation, PoissonStimulation, MixedStimulation

import torch_geometric.transforms as T
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from scipy.sparse import coo_matrix
import numpy as np

def save(x, model, w0_data, seeds, data_path, stimulation=None):
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

def calculate_isi(spikes, N, n_steps, dt=0.001) -> float:
    return N * n_steps * dt / spikes.sum()

def simulate(n_neurons, n_sims, n_steps, data_path, folder_name, max_parallel=100, firing_rate=0.1):
    """Generates a dataset"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    rng = torch.Generator().manual_seed(0)
    seeds = {
            "w0": torch.randint(0, 100000, (n_sims,), generator=rng).tolist(),
            "model": torch.randint(0, 100000, (1,), generator=rng).item(),
         }

    # Set path for saving data
    data_path = Path(data_path) / (folder_name if folder_name else Path(f"{n_neurons}_neurons_{n_sims}_datasets_{n_steps}_steps"))
    data_path.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    w0_params = GlorotParams(0, 5)
    w0_data = W0Dataset(n_neurons, n_sims, w0_params, seeds=seeds["w0"])
    data_loader = DataLoader(w0_data, batch_size=min(n_sims, max_parallel), shuffle=False)

    model_path = Path("data/saved_models") / f"{w0_params.name}_{n_neurons}_neurons_{firing_rate}_firing_rate.pt"
    model = SpikingModel(seed=seeds["model"], device=device)

    results = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        #  stim0 = RegularStimulation(targets=0, strengths=1, duration=n_steps, rates=0.2, temporal_scales=2, n_neurons=data.num_nodes, device=device)
        #  stim1 = PoissonStimulation(targets=0, strengths=1, duration=n_steps, periods=5, temporal_scales=4, n_neurons=data.num_nodes, device=device)
        #  mixed_stim = MixedStimulation([stim0, stim1])
        spikes = model.simulate(data, n_steps, stimulation=None)
        #  print(f"ISI: {calculate_isi(spikes, data.num_nodes, n_steps)}")
        print(f"Firing rate: {spikes.sum() / (n_steps * data.num_nodes):.5f}")
        results.append(spikes)

    save(results, model, w0_data, seeds, data_path)
