from spiking_network.utils import tune
import pytest

def test_parameters_require_grad(spiking_model, example_data):
    tunable_parameters = ["threshold", "alpha"]
    firing_rate = 0.1
    tune(spiking_model, example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    for parameter in tunable_parameters:
        assert spiking_model._params[parameter].requires_grad

def test_tuning_changes_parameters(spiking_model, example_data):
    from torch import allclose
    tunable_parameters = ["threshold", "alpha", "beta"]
    firing_rate = 0.1
    initial_parameters = {parameter: spiking_model._params[parameter].clone() for parameter in tunable_parameters}
    tune(spiking_model, example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    for parameter in tunable_parameters:
        assert not allclose(spiking_model._params[parameter], initial_parameters[parameter])

def test_tuning_improves_firing_rate(spiking_model, example_data, expected_firing_rate):
    from spiking_network.utils import calculate_firing_rate, simulate
    tunable_parameters = ["threshold", "alpha", "beta"]
    firing_rate = 0.1
    tune(spiking_model, example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    final_firing_rate = calculate_firing_rate(simulate(spiking_model, example_data, n_steps=1000, verbose=False))
    assert final_firing_rate > expected_firing_rate

def test_no_parameters_to_tune(spiking_model, example_data):
    from spiking_network.utils import tune
    tunable_parameters = []
    firing_rate = 0.1
    with pytest.raises(ValueError):
        tune(spiking_model, example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)

def test_save_parameters(spiking_model, example_edge_index, example_connectivity_filter, initial_state, expected_state_after_one_step):
    pass

def test_load_parameters(spiking_model, example_edge_index, example_connectivity_filter, initial_state, expected_state_after_one_step):
    pass