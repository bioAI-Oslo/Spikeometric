import numpy as np
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
from spiking_network.datasets import NormalConnectivityDataset
from spiking_network.models import GLMModel
from benchmarking.timing import time_model

def parallelization(max_neurons, n_steps, N, data_path, device="cpu"):
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    timings = []
    seed = 12345
    rng = torch.Generator().manual_seed(seed)
    p_sims = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    n_neurons = range(50, max_neurons+1, 50)
    total_sims = p_sims[-1]
    timings = np.zeros((len(p_sims), len(n_neurons)))
    for n in n_neurons:
        for i, p in enumerate(p_sims):
            w0_data = NormalConnectivityDataset.generate_examples(n, total_sims, 0, 5, rng=rng)
            model = GLMModel(rng=rng)
            data_loader = DataLoader(w0_data, batch_size=p, shuffle=False)
            time = 0
            for data in data_loader:
                data = data.to(device)
                time += time_model(model, data, n_steps, N=N)
            timings[i, n_neurons.index(n)] = time
            print(f'{n} neurons, {p} sims: {time}')

    np.savez(data_path / 'parallelization_{max_neurons}_neurons_{n_steps}_.npz', timings=timings, p_sims=p_sims)
