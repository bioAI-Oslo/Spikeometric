import pytest
from torch.testing import assert_close
import torch

def test_activation(bernoulli_glm, initial_state, example_data, expected_activation_after_one_step):
    example_connectivity_filter = bernoulli_glm.connectivity_filter(example_data.W0, example_data.edge_index)
    output = bernoulli_glm.input(example_data.edge_index, W=example_connectivity_filter, state=initial_state)
    assert_close(output, expected_activation_after_one_step)

def test_spike_probability(bernoulli_glm, expected_probability_after_one_step, expected_activation_after_one_step):
    probabilities = bernoulli_glm.non_linearity(expected_activation_after_one_step)
    assert_close(probabilities, expected_probability_after_one_step)

def test_state_after_one_step(bernoulli_glm, initial_state, example_data, expected_output_after_one_step):
    connectivity_filter = bernoulli_glm.connectivity_filter(example_data.W0, example_data.edge_index)
    state = bernoulli_glm(example_data.edge_index, connectivity_filter, state=initial_state)
    assert_close(state, expected_output_after_one_step.squeeze())

def test_connectivity_filter(bernoulli_glm, example_data, example_connectivity_filter):
    example_W0 = example_data.W0
    example_edge_index = example_data.edge_index
    W = bernoulli_glm.connectivity_filter(example_W0, example_edge_index)
    assert_close(W, example_connectivity_filter)

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
    from torch.testing import assert_close
    bernoulli_glm.add_stimulus(regular_stimulus)
    assert "stimulus.strength" in bernoulli_glm.tunable_parameters