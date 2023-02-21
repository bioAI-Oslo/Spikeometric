import pytest
from torch.testing import assert_close
import torch


# Parametrize these with the other models and their expected outputs
@pytest.mark.parametrize(
    "model,example_data,expected_input", 
    [
        (pytest.lazy_fixture('bernoulli_glm'), pytest.lazy_fixture('bernoulli_glm_network'), pytest.lazy_fixture('bernoulli_glm_expected_input')),
        (pytest.lazy_fixture('poisson_glm'), pytest.lazy_fixture('poisson_glm_network'), pytest.lazy_fixture('poisson_glm_expected_input')),
        (pytest.lazy_fixture('rectified_lnp'), pytest.lazy_fixture('rectified_lnp_network'), pytest.lazy_fixture('rectified_lnp_expected_input')),
    ],
)
def test_input(model, example_data, expected_input):
    initial_state = torch.zeros((example_data.num_nodes, model.T))
    initial_state[:, -1] = torch.randint(0, 2, (example_data.num_nodes,), generator=torch.Generator().manual_seed(14071789))
    connectivity_filter, edge_index = model.connectivity_filter(example_data.W0, example_data.edge_index)
    output = model.input(edge_index, W=connectivity_filter, state=initial_state)

    assert_close(output, expected_input)

@pytest.mark.parametrize(
    "model,example_data,expected_input",
    [
        (pytest.lazy_fixture('threshold_sam'), pytest.lazy_fixture('threshold_sam_network'), pytest.lazy_fixture('threshold_sam_expected_input')),
        (pytest.lazy_fixture('rectified_sam'), pytest.lazy_fixture('rectified_sam_network'), pytest.lazy_fixture('rectified_sam_expected_input'))
    ],
)
def test_input_input_models(model, example_data, expected_input):
    initial_state = torch.zeros((example_data.num_nodes, model.T))
    initial_state[:, -1] = torch.rand((example_data.num_nodes,), generator=torch.Generator().manual_seed(14071789))

    connectivity_filter = model.connectivity_filter(example_data.W0, example_data.edge_index)
    output = model.input(example_data.edge_index, W=connectivity_filter, state=initial_state)

    assert_close(output, expected_input)

@pytest.mark.parametrize(
    "model,expected_input,expected_rates",
    [
        (pytest.lazy_fixture('bernoulli_glm'), pytest.lazy_fixture('bernoulli_glm_expected_input'), pytest.lazy_fixture('bernoulli_glm_expected_rates')),
        (pytest.lazy_fixture('poisson_glm'), pytest.lazy_fixture('poisson_glm_expected_input'), pytest.lazy_fixture('poisson_glm_expected_rates')),
        (pytest.lazy_fixture('rectified_lnp'), pytest.lazy_fixture('rectified_lnp_expected_input'), pytest.lazy_fixture('rectified_lnp_expected_rates')),
        (pytest.lazy_fixture('threshold_sam'), pytest.lazy_fixture('threshold_sam_expected_input'), pytest.lazy_fixture('threshold_sam_expected_rates')),
        (pytest.lazy_fixture('rectified_sam'), pytest.lazy_fixture('rectified_sam_expected_input'), pytest.lazy_fixture('rectified_sam_expected_rates')),
    ],
)
def test_spike_rates(model, expected_rates, expected_input):
    probabilities = model.non_linearity(expected_input)
    assert_close(probabilities, expected_rates)

@pytest.mark.parametrize(
    "model,expected_rates,expected_output",
    [
        (pytest.lazy_fixture('bernoulli_glm'), pytest.lazy_fixture('bernoulli_glm_expected_rates'), pytest.lazy_fixture('bernoulli_glm_expected_output')),
        (pytest.lazy_fixture('poisson_glm'), pytest.lazy_fixture('poisson_glm_expected_rates'), pytest.lazy_fixture('poisson_glm_expected_output')),
        (pytest.lazy_fixture('rectified_lnp'), pytest.lazy_fixture('rectified_lnp_expected_rates'), pytest.lazy_fixture('rectified_lnp_expected_output')),
        (pytest.lazy_fixture('threshold_sam'), pytest.lazy_fixture('threshold_sam_expected_rates'), pytest.lazy_fixture('threshold_sam_expected_output')),
        (pytest.lazy_fixture('rectified_sam'), pytest.lazy_fixture('rectified_sam_expected_rates'), pytest.lazy_fixture('rectified_sam_expected_output')),
    ]
)
def test_output(model, expected_rates, expected_output):
    state = model.emit_spikes(expected_rates)
    assert_close(state, expected_output.squeeze())

@pytest.mark.parametrize(
    "model,example_data,expected_connectivity_filter",
    [
        (pytest.lazy_fixture('bernoulli_glm'), pytest.lazy_fixture('bernoulli_glm_network'), pytest.lazy_fixture('bernoulli_glm_connectivity_filter')),
        (pytest.lazy_fixture('poisson_glm'), pytest.lazy_fixture('poisson_glm_network'), pytest.lazy_fixture('poisson_glm_connectivity_filter')),
        (pytest.lazy_fixture('rectified_lnp'), pytest.lazy_fixture('rectified_lnp_network'), pytest.lazy_fixture('rectified_lnp_connectivity_filter'))
    ]
)
def test_connectivity_filter(model, example_data, expected_connectivity_filter):
    example_W0 = example_data.W0
    example_edge_index = example_data.edge_index
    W, edge_index = model.connectivity_filter(example_W0, example_edge_index)
    assert_close(W, expected_connectivity_filter)

def test_not_tunable(bernoulli_glm):
    with pytest.raises(ValueError):
        bernoulli_glm.set_tunable(["abs_ref_scale"])

def test_save_load(bernoulli_glm):
    from tempfile import NamedTemporaryFile
    from spikeometric.models import BernoulliGLM
    with NamedTemporaryFile() as f:
        bernoulli_glm.save(f.name)
        loaded_model = BernoulliGLM(1, 1, 1, 1, 1, 1, 1, 1, 1)
        loaded_model.load(f.name)
    for param, loaded_param in zip(bernoulli_glm.parameters(), loaded_model.parameters()):
        assert_close(param, loaded_param)

def test_stimulus(bernoulli_glm, regular_stimulus):
    bernoulli_glm.add_stimulus(regular_stimulus)
    assert "stimulus.strength" in bernoulli_glm.tunable_parameters

def test_fails_stimulus_is_not_callable(bernoulli_glm):
    with pytest.raises(TypeError):
        bernoulli_glm.add_stimulus(1)