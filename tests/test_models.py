import pytest
from torch.testing import assert_close

@pytest.mark.parametrize(
    "model", [pytest.lazy_fixture("spiking_model"), pytest.lazy_fixture("herman_model")]
)
def test_initialization(model, generated_dataset):
    intial_state = model.initialize_state(generated_dataset[0].num_nodes)
    assert intial_state.shape == (generated_dataset[0].num_nodes, model.time_scale)

def test_consistent_initialization(spiking_model, initial_state):
    assert_close(initial_state, spiking_model.initialize_state(initial_state.shape[0]))

def test_activation(spiking_model, example_edge_index, example_connectivity_filter, initial_state, expected_activation_after_one_step):
    output = spiking_model.activation(initial_state, example_edge_index, W=example_connectivity_filter)
    assert_close(output, expected_activation_after_one_step)

def test_spike_probability(spiking_model, expected_probability_after_one_step, expected_activation_after_one_step):
    probabilities = spiking_model.probability_of_spike(expected_activation_after_one_step)
    assert_close(probabilities, expected_probability_after_one_step)

def test_state_after_one_step(spiking_model, expected_state_after_one_step, expected_activation_after_one_step):
    probabilities = spiking_model.probability_of_spike(expected_activation_after_one_step)
    state = spiking_model.spike(probabilities)
    assert_close(state, expected_state_after_one_step)

def test_connectivity_filter(spiking_model, example_W0, example_edge_index, example_connectivity_filter):
    W = spiking_model.connectivity_filter(example_W0, example_edge_index)
    assert_close(W, example_connectivity_filter)

def test_not_tunable(spiking_model):
    with pytest.raises(ValueError):
        spiking_model.set_tunable_parameters(["abs_ref_scale"])

def test_not_a_parameter():
    from spiking_network.models import SpikingModel, HermanModel
    with pytest.raises(ValueError):
        SpikingModel(seed=0, parameters={"not_a_parameter": 0})
    with pytest.raises(ValueError):
        HermanModel(seed=0, parameters={"not_a_parameter": 0})

def test_parameter_dict(spiking_model):
    from torch import tensor
    assert spiking_model.parameter_dict == {
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

def test_save_load(spiking_model):
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile() as f:
        spiking_model.save(f.name)
        loaded_model = spiking_model.load(f.name)
    assert spiking_model.parameter_dict == loaded_model.parameter_dict

def test_stimulation(spiking_model, regular_stimulation):
    from torch.testing import assert_close
    spiking_model.add_stimulation(regular_stimulation)
    for t in range(10):
        assert_close(spiking_model.stimulate(t), regular_stimulation(t))

def test_multiple_stimulations(spiking_model, regular_stimulation, sin_stimulation):
    from torch.testing import assert_close
    spiking_model.add_stimulation([regular_stimulation, sin_stimulation])
    for t in range(10):
        model_stimulation = spiking_model.stimulate(t)
        assert_close(model_stimulation, regular_stimulation(t) + sin_stimulation(t))