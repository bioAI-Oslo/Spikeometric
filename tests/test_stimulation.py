import pytest
import torch
from torch.testing import assert_close
from torch import zeros_like

@pytest.mark.parametrize("stimulation", [pytest.lazy_fixture('regular_stimulation'), pytest.lazy_fixture('sin_stimulation'), pytest.lazy_fixture('poisson_stimulation')])
def test_zero_after_duration(stimulation):
    targets = torch.tensor([0, 4, 9])
    stim = stimulation(100, targets=targets, n_neurons=20)
    assert_close(stim, zeros_like(stim))

def test_regular_intervals(regular_stimulation):
    from torch import no_grad
    targets = torch.tensor([0, 4, 9])
    n_neurons = 20
    with no_grad():
        for i in range(2, 100):
            stim = regular_stimulation(i, targets=targets, n_neurons=n_neurons)
            indices = stim.nonzero().squeeze()
            if i % 5 <= 1:
                assert 0 in indices
                assert 4 in indices
                assert 9 in indices

def test_greater_stimulus_at_the_beginning(regular_stimulation):
    targets = torch.tensor([0, 4, 9])
    n_neurons = 20
    stim0 = regular_stimulation(5, targets, n_neurons)[0]
    stim1 = regular_stimulation(6, targets, n_neurons)[0]
    assert stim1 < stim0

def test_regular_no_negative_intervals():
    from spiking_network.stimulation import RegularStimulation
    interval = -5
    strength = 1
    temporal_scale = 2
    duration = 100
    with pytest.raises(ValueError):
        RegularStimulation(
            interval=interval,
            strength=strength,
            temporal_scale=temporal_scale,
            duration=duration,
        )

def test_sin_targets(sin_stimulation):
    targets = torch.tensor([0, 4, 9])
    stim = sin_stimulation(1, targets=targets, n_neurons=20)
    indices = stim.nonzero().squeeze()
    assert_close(indices, targets)

def test_parameter_dict(regular_stimulation):
    expected = {'decay': 0.5, 'strength': 1., 'temporal_scale': 2., 'interval': 5, 'duration': 100}
    for key, value in regular_stimulation.parameter_dict.items():
        assert value == expected[key]

def test_sin_negative_amplitudes():
    from spiking_network.stimulation import SinStimulation
    frequency = 0.01
    amplitude = -1
    duration = 100
    with pytest.raises(ValueError):
        SinStimulation(
            frequency=frequency,
            amplitude=amplitude,
            duration=duration,
        )
            
def test_sin_negative_frequencies():
    from spiking_network.stimulation import SinStimulation
    frequency = -0.01
    amplitude = 1
    duration = 100
    with pytest.raises(ValueError):
        SinStimulation(
            frequency=frequency,
            amplitude=amplitude,
            duration=duration,
        )

def test_sin_high_frequencies():
    from spiking_network.stimulation import SinStimulation
    frequency = 100
    amplitude = 1
    duration = 100
    with pytest.raises(ValueError):
        SinStimulation(
            frequency=frequency,
            amplitude=amplitude,
            duration=duration,
        )

def test_negative_durations():
    from spiking_network.stimulation import SinStimulation
    frequency = 0.01
    amplitude = 1
    duration = -100
    with pytest.raises(ValueError):
        SinStimulation(
            frequency=frequency,
            amplitude=amplitude,
            duration=duration,
        )

def test_poisson_mean_period(poisson_stimulation):
    mean_rates = poisson_stimulation._stimulation_times.sum() / poisson_stimulation.duration
    mean_periods = 1 / mean_rates
    pytest.approx(mean_periods, poisson_stimulation.interval)

def test_poisson_non_negative_period():
    from spiking_network.stimulation import PoissonStimulation
    interval = -1
    strength = 1
    duration = 100
    temporal_scale = 2
    with pytest.raises(ValueError):
        PoissonStimulation(
            interval=interval,
            strength=strength,
            duration=duration,
            temporal_scale=temporal_scale,
        )

def test_poisson_non_negative_temporal_scale():
    from spiking_network.stimulation import PoissonStimulation
    interval = 10
    strength = 1
    duration = 100
    temporal_scale = -2
    with pytest.raises(ValueError):
        PoissonStimulation(
            interval=interval,
            strength=strength,
            duration=duration,
            temporal_scale=temporal_scale,
        )

def test_poisson_stimulates_at_correct_times(poisson_stimulation):
    targets = torch.tensor([0, 4, 9])
    n_neurons = 20
    for i in range(poisson_stimulation.duration):
        stim = poisson_stimulation(i, targets=targets, n_neurons=n_neurons)
        if stim.sum() > 0:
            assert stim.nonzero().squeeze() in targets

def test_manual_stimulation(example_data, glm_model, initial_state):
    import torch
    connectivity_filter = glm_model.connectivity_filter(example_data.W0, example_data.edge_index)
    activation_without_stimulation = glm_model.activation(initial_state, example_data.edge_index, connectivity_filter, t=0)

    func = lambda t: torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    glm_model.add_stimulation(func)
    activation_with_stimulation = glm_model.activation(initial_state, example_data.edge_index, connectivity_filter, t=0)
    assert any(activation_with_stimulation != activation_without_stimulation)

def test_tuning_with_stimulation(glm_model, example_data, regular_stimulation):
    from spiking_network.utils import tune
    tunable_parameters = ["threshold", "alpha", "beta"]
    firing_rate = 0.1
    glm_model.add_stimulation(regular_stimulation)
    example_data.stimulation_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tune(glm_model, example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    assert True

def test_tune_stimulation(glm_model, example_data, regular_stimulation):
    from spiking_network.utils import tune
    tunable_parameters = ["stimulation_strength", "stimulation_decay"]
    firing_rate = 0.1
    glm_model.add_stimulation(regular_stimulation)
    example_data.stimulation_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tune(glm_model, example_data, firing_rate, tunable_parameters=tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    assert True

def test_tune_model_and_stimulation(glm_model, example_data, regular_stimulation):
    from spiking_network.utils import tune
    tunable_model_parameters = ["threshold", "alpha", "beta", "stimulation_strength", "stimulation_decay"]
    firing_rate = 0.1
    glm_model.add_stimulation(regular_stimulation)
    example_data.stimulation_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tune(glm_model, example_data, firing_rate, tunable_model_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    assert True

def test_simulate_with_stimulation(glm_model, example_data, regular_stimulation):
    from spiking_network.utils import simulate
    glm_model.add_stimulation(regular_stimulation)
    example_data.stimulation_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    simulate(glm_model, example_data, n_steps=10, verbose=False)
    assert True