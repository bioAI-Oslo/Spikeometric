import time
import numpy as np
from pathlib import Path
from spiking_network.datasets import NormalGenerator
from spiking_network.models import BernoulliGLM
from config_params import glm_params

def time_model(model, data, n_steps, N=10):
    total_time = 0
    for i in range(N):
        s = time.perf_counter()
        model.simulate(data, n_steps, verbose=False)
        e = time.perf_counter()
        total_time += e - s
    return total_time / (N * n_steps)

def time_stimulation(stimulation, n_steps, N=10):
    total_time = 0
    for i in range(N):
        s = time.perf_counter()
        for j in range(n_steps):
            stimulation(j)
        e = time.perf_counter()
        total_time += e - s
    return total_time / N

def timing(max_neurons, n_steps, N, data_path, device="cpu"):
    model = BernoulliGLM(**glm_params)
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    # Connectivity parameters
    neuron_list = range(50, max_neurons+1, 50)
    timings = []
    for n_neurons in neuron_list:
        w0_data = NormalGenerator(n_neurons, 0, 1, 5, True).generate_examples(1)
        data = w0_data[0].to(device)
        model.to(device)

        time = time_model(model, data, n_steps, N=N)

        timings.append(time)
        print(f'{n_neurons} neurons: {time}')

    np.savez(data_path / f"{max_neurons}.npz", timings=timings, neuron_list=neuron_list)