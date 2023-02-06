import pytest

def test_parameters_require_grad(bernoulli_glm, example_data):
    tunable_parameters = ["theta", "alpha"]
    firing_rate = 0.1
    bernoulli_glm.tune(example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    for parameter in tunable_parameters:
        assert bernoulli_glm.tunable_parameters[parameter].requires_grad

def test_tuning_changes_parameters(bernoulli_glm, example_data):
    from torch import allclose
    tunable_parameters = ["theta", "alpha", "beta"]
    firing_rate = 0.1
    initial_parameters = {parameter: bernoulli_glm.tunable_parameters[parameter].clone() for parameter in tunable_parameters}
    bernoulli_glm.tune(example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    for parameter in tunable_parameters:
        assert not allclose(bernoulli_glm.tunable_parameters[parameter], initial_parameters[parameter])

def test_tuning_improves_firing_rate(bernoulli_glm, example_data):
    from spiking_network.utils import calculate_firing_rate
    tunable_parameters = ["theta", "alpha", "beta"]
    firing_rate = 62.5
    initial_firing_rate = 7.2
    bernoulli_glm.tune(example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    final_firing_rate = calculate_firing_rate(bernoulli_glm.simulate(example_data, n_steps=1000, verbose=False), bernoulli_glm.dt)
    assert final_firing_rate > initial_firing_rate

def test_no_parameters_to_tune(bernoulli_glm, example_data):
    tunable_parameters = []
    firing_rate = 0.1
    with pytest.raises(ValueError):
        bernoulli_glm.tune(example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)