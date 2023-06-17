import torch
from torch_geometric.loader import DataLoader
import pytest

@pytest.mark.parametrize("stimulus", [pytest.lazy_fixture("regular_stimulus"), pytest.lazy_fixture("poisson_stimulus"), pytest.lazy_fixture("loaded_stimulus")])
def test_simulates_on_gpu_if_available(bernoulli_glm, example_data, stimulus):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bernoulli_glm.add_stimulus(stimulus)
    bernoulli_glm.to(device)
    example_data.to(device)
    spikes = bernoulli_glm.simulate(example_data, 10, verbose=False)
    assert spikes.is_cuda == torch.cuda.is_available()

@pytest.mark.parametrize("stimulus", [pytest.lazy_fixture("regular_stimulus"), pytest.lazy_fixture("poisson_stimulus"), pytest.lazy_fixture("loaded_stimulus")])
def test_all_tensors_on_same_device(bernoulli_glm, example_data, stimulus):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bernoulli_glm.add_stimulus(stimulus)
    bernoulli_glm.to(device)
    example_data.to(device)
    for parameter in bernoulli_glm.state_dict():
        assert bernoulli_glm.state_dict()[parameter].is_cuda == torch.cuda.is_available()
    assert example_data.W0.is_cuda == torch.cuda.is_available()
    assert example_data.edge_index.is_cuda == torch.cuda.is_available()
