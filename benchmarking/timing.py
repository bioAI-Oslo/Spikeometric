import sys
import torch
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path

from spiking_network.datasets import W0Dataset, GlorotParams
from spiking_network.models import SpikingModel


def time_model(model, data, n_steps, N=10):
    total_time = 0
    for i in range(N):
        s = time.perf_counter()
        model.simulate(data, n_steps, verbose=False)
        e = time.perf_counter()
        total_time += e - s
    return total_time / N

def timing(max_neurons, n_steps, N, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dist_params = GlorotParams(0, 5)
    model = SpikingModel(seed=0, device=device)
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    # Connectivity parameters
    neuron_list = range(500, max_neurons+1, 500)
    timings = []
    for n_neurons in neuron_list:
        w0_data = W0Dataset(n_neurons, 1, dist_params, seeds=0)
        data = w0_data[0].to(device)

        time = time_model(model, data, n_steps, N=N)

        timings.append(time)
        print(f'{n_neurons} neurons: {time}')

    np.savez(data_path / f"{max_neurons}.npz", timings=timings, neuron_list=neuron_list)

if __name__ == '__main__':
    main()