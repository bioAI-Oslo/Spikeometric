import pytest
from torch.testing import assert_close
import torch

@pytest.mark.parametrize(
    "model", [pytest.lazy_fixture("glm_model"), pytest.lazy_fixture("lnp_model")]
)
def test_initialization(model, saved_glorot_dataset):
    intial_state = model.initialize_state(saved_glorot_dataset[0].num_nodes)
    assert intial_state.shape == (saved_glorot_dataset[0].num_nodes, model.time_scale)

def test_consistent_initialization(glm_model, initial_state):
    assert_close(initial_state, glm_model.initialize_state(initial_state.shape[0]))

def test_activation(glm_model, example_data, expected_activation_after_one_step):
    initial_state = glm_model.initialize_state(example_data.num_nodes)
    example_connectivity_filter = glm_model.connectivity_filter(example_data.W0, example_data.edge_index)
    output = glm_model.activation(initial_state, example_data.edge_index, W=example_connectivity_filter)
    assert_close(output, expected_activation_after_one_step)

def test_spike_probability(glm_model, expected_probability_after_one_step, expected_activation_after_one_step):
    probabilities = glm_model.probability_of_spike(expected_activation_after_one_step)
    assert_close(probabilities, expected_probability_after_one_step)

def test_state_after_one_step(glm_model, example_data, expected_state_after_one_step):
    connectivity_filter = glm_model.connectivity_filter(example_data.W0, example_data.edge_index)
    initial_state = glm_model.initialize_state(example_data.num_nodes)
    state = glm_model(initial_state, example_data.edge_index, connectivity_filter)
    assert_close(state, expected_state_after_one_step)

def test_connectivity_filter(glm_model, example_data, example_connectivity_filter):
    example_W0 = example_data.W0
    example_edge_index = example_data.edge_index
    W = glm_model.connectivity_filter(example_W0, example_edge_index)
    assert_close(W, example_connectivity_filter)

def test_not_tunable(glm_model):
    with pytest.raises(ValueError):
        glm_model.set_tunable_parameters(["abs_ref_scale"])

def test_not_a_parameter():
    from spiking_network.models import GLMModel, LNPModel
    with pytest.raises(ValueError):
        GLMModel(seed=0, parameters={"not_a_parameter": 0})
    with pytest.raises(ValueError):
        LNPModel(seed=0, parameters={"not_a_parameter": 0})

def test_parameter_dict(glm_model):
    from torch import tensor
    assert glm_model.parameter_dict == {
            "alpha": tensor(0.2),
            "beta": tensor(0.5),
            "threshold": tensor(5.),
            "abs_ref_strength": tensor(-100.),
            "rel_ref_strength": tensor(-30.),
            "abs_ref_scale": tensor(3),
            "rel_ref_scale": tensor(7),
            "influence_scale": tensor(5),
            "time_scale": tensor(10)
    }

def test_save_load(glm_model):
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile() as f:
        glm_model.save(f.name)
        loaded_model = glm_model.load(f.name)
    assert glm_model.parameter_dict == loaded_model.parameter_dict

def test_stimulation(glm_model, regular_stimulation):
    from torch.testing import assert_close
    glm_model.add_stimulation(regular_stimulation)
    for t in range(10):
        assert_close(glm_model.stimulate(t), regular_stimulation(t))

def test_multiple_stimulations(glm_model, regular_stimulation, sin_stimulation):
    from torch.testing import assert_close
    glm_model.add_stimulation([regular_stimulation, sin_stimulation])
    for t in range(10):
        model_stimulation = glm_model.stimulate(t)
        assert_close(model_stimulation, regular_stimulation(t) + sin_stimulation(t))