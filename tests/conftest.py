import pytest
import torch

# Dataset fixtures
@pytest.fixture 
def generated_glorot_data():
    from spiking_network.datasets import NormalGenerator
    n_neurons = 20
    n_networks = 10
    rng = torch.Generator().manual_seed(14071789)
    generator = NormalGenerator(n_neurons, mean=0, std=5, glorot=True, rng=rng)
    data = generator.generate_examples(n_networks)
    return data

@pytest.fixture
def generated_normal_data():
    from spiking_network.datasets import NormalGenerator
    n_neurons = 100
    n_networks = 10
    rng = torch.Generator().manual_seed(14071789)
    generator = NormalGenerator(n_neurons, mean=0, std=1, rng=rng)
    data = generator.generate_examples(n_networks)
    return data

@pytest.fixture
def generated_uniform_data():
    from spiking_network.datasets import UniformGenerator
    import shutil
    n_neurons = 20
    n_sims = 10
    rng = torch.Generator().manual_seed(14071789)
    low = -0.002289225919299652
    high = 0
    generator = UniformGenerator(n_neurons, low, high, sparsity=0.9, rng=rng)
    data = generator.generate_examples(n_sims)
    return data

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
def example_connectivity_filter():
    connectivity_filter = torch.load("tests/test_data/example_connectivity_filter.pt")
    return connectivity_filter

# Model and simulation fixtures
@pytest.fixture
def bernoulli_glm():
    from spiking_network.models import BernoulliGLM
    rng = torch.Generator().manual_seed(14071789)
    model = BernoulliGLM(
        theta=5.,
        dt=1,
        coupling_window=5,
        abs_ref_scale=3,
        abs_ref_strength=-100.,
        beta=0.5,
        rel_ref_scale=7,
        rel_ref_strength=-30.,
        alpha=0.2,
        rng=rng,
    )
    return model

@pytest.fixture
def threshold_sam():
    from spiking_network.models import ThresholdSAM
    rng = torch.Generator().manual_seed(14071789)
    model = ThresholdSAM(
        r=0.025,
        b=0.001,
        tau=10.,
        dt=0.1,
        sigma=0.3,
        rho=0.07,
        theta=1.378e-3,
        rng=rng,
    )
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
def expected_output_after_one_step():
    import torch
    expected_state = torch.load("tests/test_data/expected_output_after_one_step.pt")

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
    interval = 100
    strength = 5.
    tau = 10
    n_events = 10
    stimulation = RegularStimulation(
        strength=strength,
        interval=interval,
        n_events=n_events,
        tau=tau,
    )
    return stimulation

@pytest.fixture
def sin_stimulation():
    from spiking_network.stimulation import SinStimulation
    amplitude = 2.
    period = 10
    duration = 100
    stimulation = SinStimulation(
        amplitude=amplitude,
        period=period,
        duration=duration,
    )
    return stimulation

@pytest.fixture
def poisson_stimulation():
    from spiking_network.stimulation import PoissonStimulation
    mean_interval = 100
    strength = 5.
    n_events = 10
    tau = 10
    stimulation = PoissonStimulation(
        strength=strength,
        mean_interval=mean_interval,
        n_events=n_events,
        tau=tau,
    )
    return stimulation