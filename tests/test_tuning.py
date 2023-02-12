import pytest
from torch_geometric.loader import DataLoader

def test_only_stimulus_tunable(bernoulli_glm, sin_stimulus, saved_glorot_dataset):
    bernoulli_glm.add_stimulus(sin_stimulus)
    data_loader = DataLoader(saved_glorot_dataset, batch_size=10)
    initial_parameters = {
        parameter: bernoulli_glm.tunable_parameters[parameter].clone()
        for parameter in bernoulli_glm.tunable_parameters
        if parameter.startswith("stimulus")
    }
    for data in data_loader:
        bernoulli_glm.tune(data, 10, tunable_parameters="stimulus", n_steps=100, n_epochs=1, lr=0.01, verbose=False)
        for parameter in initial_parameters:
            assert not bernoulli_glm.tunable_parameters[parameter].requires_grad
            assert initial_parameters[parameter] == bernoulli_glm.tunable_parameters[parameter]
        
def test_only_model_tunable(bernoulli_glm, sin_stimulus, saved_glorot_dataset):
    bernoulli_glm.add_stimulus(sin_stimulus)
    data_loader = DataLoader(saved_glorot_dataset, batch_size=10)
    initial_parameters = {
        parameter: bernoulli_glm.tunable_parameters[parameter].clone()
        for parameter in bernoulli_glm.tunable_parameters
        if not parameter.startswith("stimulus")
    }
    for data in data_loader:
        bernoulli_glm.tune(data, 10, tunable_parameters="model", n_steps=100, n_epochs=1, lr=0.01, verbose=False)
        for parameter in initial_parameters:
            assert not bernoulli_glm.tunable_parameters[parameter].requires_grad
            assert not initial_parameters[parameter] == bernoulli_glm.tunable_parameters[parameter]

def test_tune_rectified_sa_model(rectified_sam, rectified_sam_network):
    initial_spikes = rectified_sam.simulate(rectified_sam_network, 100)
    rectified_sam.tune(rectified_sam_network, 10, n_steps=100, n_epochs=1, lr=0.01, verbose=False)
    final_spikes = rectified_sam.simulate(rectified_sam_network, 100)
    assert not initial_spikes.float().mean() == final_spikes.float().mean()

def test_all_parameters_tunable(bernoulli_glm, saved_glorot_dataset):
    data_loader = DataLoader(saved_glorot_dataset, batch_size=10)
    initial_parameters = {parameter: bernoulli_glm.tunable_parameters[parameter].clone() for parameter in bernoulli_glm.tunable_parameters}
    for data in data_loader:
        bernoulli_glm.tune(data, 10, n_steps=100, n_epochs=1, lr=0.01, verbose=False)
        for parameter in initial_parameters:
            assert not bernoulli_glm.tunable_parameters[parameter].requires_grad
            assert not initial_parameters[parameter] == bernoulli_glm.tunable_parameters[parameter]

def test_parameters_require_grad(bernoulli_glm, example_data):
    tunable_parameters = ["theta", "alpha"]
    firing_rate = 10
    bernoulli_glm.tune(example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    for parameter in tunable_parameters:
        assert not bernoulli_glm.tunable_parameters[parameter].requires_grad

def test_tuning_changes_parameters(bernoulli_glm, example_data):
    tunable_parameters = ["theta", "alpha", "beta"]
    firing_rate = 10
    initial_parameters = {parameter: bernoulli_glm.tunable_parameters[parameter].clone() for parameter in tunable_parameters}
    bernoulli_glm.tune(example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    for parameter in tunable_parameters:
        assert not bernoulli_glm.tunable_parameters[parameter] == initial_parameters[parameter]

def test_tuning_improves_firing_rate(bernoulli_glm, example_data):
    tunable_parameters = ["theta", "alpha", "beta"]
    firing_rate = 62.5
    initial_firing_rate = 7.2
    bernoulli_glm.tune(example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    X = bernoulli_glm.simulate(example_data, n_steps=1000, verbose=False)
    final_firing_rate = X.float().mean() / bernoulli_glm.dt * 1000
    assert final_firing_rate > initial_firing_rate

def test_no_parameters_to_tune(bernoulli_glm, example_data):
    tunable_parameters = []
    firing_rate = 10
    with pytest.raises(ValueError):
        bernoulli_glm.tune(example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)