import pytest

# Dataset fixtures
@pytest.fixture 
def generated_glorot_dataset():
    from spiking_network.datasets import NormalConnectivityDataset, GlorotParams
    import shutil
    n_neurons = 20
    n_sims = 10
    params = GlorotParams(0, 5)
    dataset = NormalConnectivityDataset(n_neurons, n_sims, params, seed=14071789, root="tests/test_data/generated_glorot_dataset")
    shutil.rmtree("tests/test_data/generated_glorot_dataset")
    return dataset

@pytest.fixture
def generated_normal_dataset():
    from spiking_network.datasets import NormalConnectivityDataset, NormalParams
    import shutil as shuntil
    n_neurons = 100
    n_sims = 10
    params = NormalParams(0, 1)
    dataset = NormalConnectivityDataset(n_neurons, n_sims, params, seed=14071789, root="tests/test_data/generated_normal_dataset")
    shuntil.rmtree("tests/test_data/generated_normal_dataset")
    return dataset

@pytest.fixture
def generated_uniform_dataset():
    from spiking_network.datasets import UniformConnectivityDataset
    import shutil
    n_neurons = 20
    n_sims = 10
    dataset = UniformConnectivityDataset(n_neurons, n_sims, seed=14071789, root="tests/test_data/generated_uniform_dataset", sparsity=0.5)
    shutil.rmtree("tests/test_data/generated_uniform_dataset")
    return dataset

@pytest.fixture
def saved_glorot_dataset():
    from spiking_network.datasets import ConnectivityDataset
    dataset = ConnectivityDataset(root="tests/test_data/example_glorot_dataset")
    return dataset

@pytest.fixture
def sparse_glorot_dataset():
    from spiking_network.datasets import ConnectivityDataset
    dataset = ConnectivityDataset(root="tests/test_data/example_sparse_glorot_dataset")
    return dataset

@pytest.fixture
def data_loader(saved_glorot_dataset):
    from torch.utils.data import DataLoader
    data_loader = DataLoader(saved_glorot_dataset, batch_size=10)
    return data_loader

@pytest.fixture
def example_data(saved_glorot_dataset):
    return saved_glorot_dataset[0]

@pytest.fixture
def example_uniform_data():
    from spiking_network.datasets import UniformConnectivityDataset
    n_neurons = 100
    n_sims = 1
    dataset = UniformConnectivityDataset(n_neurons, n_sims, seed=14071789, root="tests/test_data/example_uniform_dataset")
    return dataset[0]

@pytest.fixture
def example_connectivity_filter():
    import torch
    connectivity_filter = torch.load("tests/test_data/example_connectivity_filter.pt")
    return connectivity_filter

# Model and simulation fixtures
@pytest.fixture
def glm_model():
    from spiking_network.models import GLMModel
    model = GLMModel(seed=14071789)
    return model

@pytest.fixture
def lnp_model():
    from spiking_network.models import LNPModel
    model = LNPModel(seed=14071789)
    return model

@pytest.fixture
def initial_state():
    import torch
    initial_state = torch.load("tests/test_data/initial_state.pt")
    return initial_state

@pytest.fixture
def expected_activation_after_one_step():
    import torch
    expected_activation = torch.load("tests/test_data/expected_activation_after_one_step.pt")
    return expected_activation

@pytest.fixture
def expected_probability_after_one_step():
    import torch
    expected_probability = torch.load("tests/test_data/expected_probability_after_one_step.pt")
    return expected_probability

@pytest.fixture
def expected_state_after_one_step():
    import torch
    expected_state = torch.load("tests/test_data/expected_state_after_one_step.pt")
    return expected_state

@pytest.fixture
def expected_output_after_ten_steps():
    import torch
    expected_output = torch.load("tests/test_data/expected_output_after_ten_steps.pt")
    return expected_output

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