import pytest

# Dataset fixtures
@pytest.fixture 
def generated_dataset():
    from spiking_network.datasets import W0Dataset, GlorotParams
    import shutil
    n_neurons = 20
    n_sims = 10
    params = GlorotParams(0, 5)
    dataset = W0Dataset(n_neurons, n_sims, params, seed=0, root="tests/test_data/generated_glorot_w0")
    shutil.rmtree("tests/test_data/generated_glorot_w0")
    return dataset

@pytest.fixture
def generated_normal_dataset():
    from spiking_network.datasets import W0Dataset, NormalParams
    import shutil as shuntil
    n_neurons = 100
    n_sims = 10
    params = NormalParams(0, 1)
    dataset = W0Dataset(n_neurons, n_sims, params, seed=0, root="tests/test_data/generated_normal_w0")
    shuntil.rmtree("tests/test_data/generated_normal_w0")
    return dataset

@pytest.fixture
def generated_mexican_hat_dataset():
    from spiking_network.datasets import MexicanHatDataset
    import shutil
    n_neurons = 20
    n_sims = 10
    dataset = MexicanHatDataset(n_neurons, n_sims, seed=0, root="tests/test_data/generated_mexican_hat")
    shutil.rmtree("tests/test_data/generated_mexican_hat")
    return dataset

@pytest.fixture
def saved_dataset():
    from spiking_network.datasets import ConnectivityDataset
    dataset = ConnectivityDataset(root="tests/test_data/example_dataset")
    return dataset

@pytest.fixture
def sparse_dataset():
    from spiking_network.datasets import W0Dataset, GlorotParams
    dataset = W0Dataset(20, 10, distribution_params=GlorotParams(0, 5), seed=0, root="tests/test_data/sparse_glorot_w0", sparsity=0.5)
    return dataset

@pytest.fixture
def data_loader(generated_dataset):
    from torch.utils.data import DataLoader
    data_loader = DataLoader(generated_dataset, batch_size=10)
    return data_loader

@pytest.fixture
def example_data():
    import numpy as np
    import torch
    from torch_geometric.data import Data
    data = np.load("tests/test_data/example_data.npz")
    W0 = torch.from_numpy(data["example_W0"])
    edge_index = torch.from_numpy(data["example_edge_index"])
    num_nodes = data["num_nodes"].item()
    example_data = Data(W0=W0, edge_index=edge_index, num_nodes=num_nodes)
    return example_data

@pytest.fixture
def example_mexican_hat_data():
    from spiking_network.datasets import MexicanHatDataset
    n_neurons = 100
    n_sims = 1
    dataset = MexicanHatDataset(n_neurons, n_sims, seed=0, root="tests/test_data/mexican_hat_dataset")
    return dataset[0]

@pytest.fixture
def example_W0():
    import numpy as np
    import torch
    w0 = torch.from_numpy(np.load("tests/test_data/example_data.npz")["example_W0"])
    return w0

@pytest.fixture
def example_edge_index():
    import numpy as np
    import torch
    edge_index = torch.from_numpy(np.load("tests/test_data/example_data.npz")["example_edge_index"])
    return edge_index

@pytest.fixture
def example_connectivity_filter():
    import numpy as np
    import torch
    connectivity_filter = torch.from_numpy(np.load("tests/test_data/example_connectivity_filter.npz")["connectivity_filter"])
    return connectivity_filter

# Model and simulation fixtures
@pytest.fixture
def spiking_model():
    from spiking_network.models import SpikingModel
    model = SpikingModel(seed=0)
    return model

@pytest.fixture
def mexican_model():
    from spiking_network.models import MexicanModel
    model = MexicanModel(seed=0)
    return model

@pytest.fixture
def initial_state():
    import torch
    import numpy as np
    initial_state = torch.from_numpy(np.load("tests/test_data/initial_state.npz")["initial_state"])
    return initial_state

@pytest.fixture
def expected_activation_after_one_step():
    import torch
    import numpy as np
    expected_output = torch.from_numpy(np.load("tests/test_data/expected_activation_after_one_step.npz")["expected_activation_after_one_step"])
    return expected_output

@pytest.fixture
def expected_probability_after_one_step():
    import torch
    import numpy as np
    expected_probability = torch.from_numpy(np.load("tests/test_data/expected_probability_after_one_step.npz")["expected_probability_after_one_step"])
    return expected_probability

@pytest.fixture
def expected_state_after_one_step():
    import torch
    import numpy as np
    expected_state = torch.from_numpy(np.load("tests/test_data/expected_state_after_one_step.npz")["expected_state_after_one_step"])
    return expected_state

@pytest.fixture
def expected_output_after_ten_steps():
    import torch
    import numpy as np
    expected_output = torch.from_numpy(np.load("tests/test_data/expected_output_after_ten_steps.npz")["expected_output_after_ten_steps"])
    return expected_output

@pytest.fixture
def expected_firing_rate():
    import torch
    import numpy as np
    expected_firing_rate = torch.from_numpy(np.load("tests/test_data/expected_firing_rate.npz")["expected_firing_rate"])
    return expected_firing_rate

@pytest.fixture
def time_to_simulate_100_steps():
    import torch
    import numpy as np
    time_to_simulate_100_steps = torch.from_numpy(np.load("tests/test_data/time_to_simulate_100_steps.npz")["time_to_simulate_100_steps"]).item()
    return time_to_simulate_100_steps

# Stimulation fixtures
@pytest.fixture
def regular_stimulation():
    from spiking_network.stimulation import RegularStimulation
    targets = [0, 4, 9]
    intervals = [5, 7, 9]
    strengths = 1
    temporal_scale = 2
    durations = 100
    stimulation = RegularStimulation(
        targets=targets,
        intervals=intervals,
        strengths=strengths,
        temporal_scale=temporal_scale,
        durations=durations,
        total_neurons=20,
    )
    return stimulation

@pytest.fixture
def sin_stimulation():
    from spiking_network.stimulation import SinStimulation
    targets = [0, 4, 9]
    amplitudes = 2
    frequencies = 0.1
    durations = 100
    total_neurons = 20
    stimulation = SinStimulation(
        targets=targets,
        amplitudes=amplitudes,
        frequencies=frequencies,
        durations=durations,
        total_neurons=total_neurons,
    )
    return stimulation

@pytest.fixture
def poisson_stimulation():
    from spiking_network.stimulation import PoissonStimulation
    targets = [0, 4, 9]
    intervals = 3
    strengths = 1
    durations = 100
    temporal_scale = 1
    total_neurons = 20
    stimulation = PoissonStimulation(
        targets=targets,
        intervals=intervals,
        strengths=strengths,
        temporal_scale=temporal_scale,
        durations=durations,
        total_neurons=total_neurons,
    )
    return stimulation