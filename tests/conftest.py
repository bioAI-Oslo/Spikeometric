import pytest
import torch

# Dataset fixtures
@pytest.fixture 
def generated_glorot_data():
    from spikeometric.datasets import NormalGenerator
    n_neurons = 20
    n_networks = 10
    rng = torch.Generator().manual_seed(14071789)
    generator = NormalGenerator(n_neurons, mean=0, std=5, glorot=True, rng=rng)
    data = generator.generate(n_networks)
    return data

@pytest.fixture
def generated_normal_data():
    from spikeometric.datasets import NormalGenerator
    n_neurons = 100
    n_networks = 10
    rng = torch.Generator().manual_seed(14071789)
    generator = NormalGenerator(n_neurons, mean=0, std=1, rng=rng)
    data = generator.generate(n_networks)
    return data

@pytest.fixture
def generated_uniform_data():
    from spikeometric.datasets import UniformGenerator
    n_neurons = 20
    n_sims = 10
    rng = torch.Generator().manual_seed(14071789)
    low = -0.002289225919299652
    high = 0
    generator = UniformGenerator(n_neurons, low, high, sparsity=0.9, rng=rng)
    data = generator.generate(n_sims)
    return data

@pytest.fixture
def generated_mexican_hat_data():
    from spikeometric.datasets import MexicanHatGenerator
    n_neurons = 20
    n_sims = 10
    rng = torch.Generator().manual_seed(14071789)
    generator = MexicanHatGenerator(n_neurons, a=1.0015, sigma_1=6.98, sigma_2=7.)
    data = generator.generate(n_sims)
    return data

@pytest.fixture
def saved_glorot_dataset():
    from spikeometric.datasets import ConnectivityDataset
    dataset = ConnectivityDataset(root="tests/test_data/example_glorot_dataset")
    return dataset

@pytest.fixture
def sparse_glorot_dataset():
    from spikeometric.datasets import ConnectivityDataset
    dataset = ConnectivityDataset(root="tests/test_data/example_sparse_glorot_dataset")
    return dataset

@pytest.fixture
def example_data(saved_glorot_dataset):
    return saved_glorot_dataset[0]

@pytest.fixture
def data_with_stimulus_mask(saved_glorot_dataset):
    data = saved_glorot_dataset[0]
    data.stimulus_mask = torch.isin(torch.arange(20), torch.randperm(20)[:10])
    return data

@pytest.fixture
def data_loader(saved_glorot_dataset):
    from torch.utils.data import DataLoader
    data_loader = DataLoader(saved_glorot_dataset, batch_size=10)
    return data_loader


@pytest.fixture
def bernoulli_glm_connectivity_filter():
    connectivity_filter = torch.load("tests/test_data/connectivity_filter/bernoulli_glm_connectivity_filter.pt")
    return connectivity_filter

@pytest.fixture
def poisson_glm_connectivity_filter():
    connectivity_filter = torch.load("tests/test_data/connectivity_filter/poisson_glm_connectivity_filter.pt")
    return connectivity_filter

@pytest.fixture
def rectified_lnp_connectivity_filter():
    connectivity_filter = torch.load("tests/test_data/connectivity_filter/rectified_lnp_connectivity_filter.pt")
    return connectivity_filter

# Model and simulation fixtures
@pytest.fixture
def bernoulli_glm():
    from spikeometric.models import BernoulliGLM
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
    from spikeometric.models import ThresholdSAM
    rng = torch.Generator().manual_seed(14071789)
    model = ThresholdSAM(
        r=0.025,
        b=0.001,
        tau=10,
        dt=0.1,
        sigma=0.3,
        rho=0.07,
        theta=1.378e-3,
        rng=rng,
    )
    return model

@pytest.fixture
def poisson_glm():
    from spikeometric.models import PoissonGLM
    rng = torch.Generator().manual_seed(14071789)
    model = PoissonGLM(
        alpha=15.9,
        beta=10,
        dt=0.1,
        T=200,
        tau=10,
        r=0.025,
        b=0.001,
        rng=rng,
    )
    return model

@pytest.fixture
def rectified_lnp():
    from spikeometric.models import RectifiedLNP
    rng = torch.Generator().manual_seed(14071789)
    model = RectifiedLNP(
        lambda_0=9.8,
        theta=-0.002,
        dt=0.1,
        T=200,
        tau=10.,
        r=0.025,
        b=0.001,
        rng=rng,
    )
    return model

@pytest.fixture
def rectified_sam():
    from spikeometric.models import RectifiedSAM
    rng = torch.Generator().manual_seed(14071789)
    model = RectifiedSAM(
        lambda_0=1.0861,
        theta=-1.37e-2,
        tau=10.,
        dt=0.1,
        r=0.025,
        b=0.001,
        rng=rng,
    )
    return model

@pytest.fixture
def bernoulli_glm_network():
    from spikeometric.datasets import NormalGenerator
    network = NormalGenerator(20, mean=0, std=5, glorot=True, rng=torch.Generator().manual_seed(14071789)).generate(1, add_self_loops=True)[0]
    return network

@pytest.fixture
def bernoulli_glm_expected_input():
    expected_input = torch.load("tests/test_data/expected_input/bernoulli_glm_expected_input.pt")
    return expected_input

@pytest.fixture
def bernoulli_glm_expected_rates():
    expected_rates = torch.load("tests/test_data/expected_rates/bernoulli_glm_expected_rates.pt")
    return expected_rates

@pytest.fixture
def bernoulli_glm_expected_output():
    expected_state = torch.load("tests/test_data/expected_output/bernoulli_glm_expected_output.pt")
    return expected_state

@pytest.fixture
def poisson_glm_network():
    from spikeometric.datasets import NormalGenerator
    network = NormalGenerator(20, mean=0, std=1, glorot=True, rng=torch.Generator().manual_seed(14071789)).generate(1)[0]
    return network

@pytest.fixture
def poisson_glm_expected_input():
    return torch.load("tests/test_data/expected_input/poisson_glm_expected_input.pt")

@pytest.fixture
def poisson_glm_expected_rates():
    expected_rates = torch.load("tests/test_data/expected_rates/poisson_glm_expected_rates.pt")
    return expected_rates

@pytest.fixture
def poisson_glm_expected_output():
    expected_state = torch.load("tests/test_data/expected_output/poisson_glm_expected_output.pt")
    return expected_state

@pytest.fixture
def rectified_lnp_network():
    from spikeometric.datasets import NormalGenerator
    network = NormalGenerator(20, mean=0, std=1, glorot=True, rng=torch.Generator().manual_seed(14071789)).generate(1)[0]
    return network

@pytest.fixture
def rectified_lnp_expected_input():
    return torch.load("tests/test_data/expected_input/rectified_lnp_expected_input.pt")

@pytest.fixture
def rectified_lnp_expected_rates():
    expected_rates = torch.load("tests/test_data/expected_rates/rectified_lnp_expected_rates.pt")
    return expected_rates

@pytest.fixture
def rectified_lnp_expected_output():
    expected_state = torch.load("tests/test_data/expected_output/rectified_lnp_expected_output.pt")
    return expected_state

@pytest.fixture
def threshold_sam_network():
    from spikeometric.datasets import UniformGenerator
    network = UniformGenerator(20, low=-0.002289, high=0, rng=torch.Generator().manual_seed(14071789), sparsity=0.9).generate(1)[0]
    return network

@pytest.fixture
def threshold_sam_expected_input():
    return torch.load("tests/test_data/expected_input/threshold_sam_expected_input.pt")

@pytest.fixture
def threshold_sam_expected_rates():
    expected_rates = torch.load("tests/test_data/expected_rates/threshold_sam_expected_rates.pt")
    return expected_rates

@pytest.fixture
def threshold_sam_expected_output():
    expected_state = torch.load("tests/test_data/expected_output/threshold_sam_expected_output.pt")
    return expected_state

@pytest.fixture
def rectified_sam_network():
    from spikeometric.datasets import UniformGenerator
    network = UniformGenerator(20, low=-0.002289, high=0, rng=torch.Generator().manual_seed(14071789), sparsity=0.9).generate(1)[0]
    return network

@pytest.fixture
def rectified_sam_expected_input():
    return torch.load("tests/test_data/expected_input/rectified_sam_expected_input.pt")

@pytest.fixture
def rectified_sam_expected_rates():
    expected_rates = torch.load("tests/test_data/expected_rates/rectified_sam_expected_rates.pt")
    return expected_rates

@pytest.fixture
def rectified_sam_expected_output():
    expected_state = torch.load("tests/test_data/expected_output/rectified_sam_expected_output.pt")
    return expected_state

@pytest.fixture
def expected_output_after_ten_steps():
    import torch
    expected_output = torch.load("tests/test_data/expected_output_after_ten_steps.pt")
    return expected_output

# Stimulus fixtures
@pytest.fixture
def regular_stimulus():
    from spikeometric.stimulus import RegularStimulus
    period = 100
    strength = 5.
    tau = 10
    stop = 1000
    stimulus = RegularStimulus(
        strength=strength,
        period=period,
        tau=tau,
        stop=stop,
    )
    return stimulus

@pytest.fixture
def sin_stimulus():
    from spikeometric.stimulus import SinStimulus
    amplitude = 2.
    period = 10
    duration = 100
    stimulus = SinStimulus(
        amplitude=amplitude,
        period=period,
        duration=duration,
    )
    return stimulus

@pytest.fixture
def poisson_stimulus():
    from spikeometric.stimulus import PoissonStimulus
    mean_interval = 100
    strength = 5.
    tau = 10
    stimulus = PoissonStimulus(
        strength=strength,
        mean_interval=mean_interval,
        duration=1000,
        tau=tau,
    )
    return stimulus