from spiking_network.utils import tune
import pytest

def test_parameters_require_grad(glm_model, example_data):
    tunable_parameters = ["threshold", "alpha"]
    firing_rate = 0.1
    tune(glm_model, example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    for parameter in tunable_parameters:
        assert glm_model._tunable_params[parameter].requires_grad

def test_tuning_changes_parameters(glm_model, example_data):
    from torch import allclose
    tunable_parameters = ["threshold", "alpha", "beta"]
    firing_rate = 0.1
    initial_parameters = {parameter: glm_model._tunable_params[parameter].clone() for parameter in tunable_parameters}
    tune(glm_model, example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    for parameter in tunable_parameters:
        assert not allclose(glm_model._tunable_params[parameter], initial_parameters[parameter])

def test_tuning_improves_firing_rate(glm_model, example_data):
    from spiking_network.utils import calculate_firing_rate, simulate
    tunable_parameters = ["threshold", "alpha", "beta"]
    firing_rate = 0.1
    initial_firing_rate = 0.0072
    tune(glm_model, example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    final_firing_rate = calculate_firing_rate(simulate(glm_model, example_data, n_steps=1000, verbose=False))
    assert final_firing_rate > initial_firing_rate

def test_no_parameters_to_tune(glm_model, example_data):
    from spiking_network.utils import tune
    tunable_parameters = []
    firing_rate = 0.1
    with pytest.raises(ValueError):
        tune(glm_model, example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)