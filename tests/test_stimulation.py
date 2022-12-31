import pytest

def test_regular_targets(regular_stimulation):
    from torch.testing import assert_close
    stim = regular_stimulation(0)
    indices = stim.nonzero().squeeze()
    assert_close(indices, regular_stimulation.targets)

@pytest.mark.parametrize("stimulation", [pytest.lazy_fixture('regular_stimulation'), pytest.lazy_fixture('sin_stimulation')])
def test_zero_after_duration(stimulation):
    from torch.testing import assert_close
    from torch import zeros_like
    stim = stimulation(100)
    assert_close(stim, zeros_like(stim))

def test_regular_intervals(regular_stimulation):
    from torch import no_grad
    with no_grad():
        for i in range(0, 100):
            stim = regular_stimulation(i)
            indices = stim.nonzero().squeeze()
            if i % 5 <= 1:
                assert 0 in indices
            if i % 7 <= 1:
                assert 4 in indices
            if i % 9 <= 1:
                assert 9 in indices

def test_regular_no_negative_intervals(regular_stimulation):
    from spiking_network.stimulation import RegularStimulation
    targets = [0, 4, 9]
    intervals = -5
    strengths = 1
    temporal_scales = 2
    durations = [50, 70, 100]
    with pytest.raises(ValueError):
        RegularStimulation(
            targets=targets,
            intervals=intervals,
            strengths=strengths,
            temporal_scales=temporal_scales,
            durations=durations,
            total_neurons=20,
        )

def test_regular_equal_length(regular_stimulation):
    from spiking_network.stimulation import RegularStimulation
    targets = [0, 4, 9]
    intervals = [5, 7, 9, 10]
    strengths = 1
    temporal_scales = 2
    durations = 100
    with pytest.raises(ValueError):
        RegularStimulation(
            targets=targets,
            intervals=intervals,
            strengths=strengths,
            temporal_scales=temporal_scales,
            durations=durations,
            total_neurons=20,
        )

def test_sin_targets(sin_stimulation):
    from torch.testing import assert_close
    stim = sin_stimulation(1)
    indices = stim.nonzero().squeeze()
    assert_close(indices, sin_stimulation.targets)

def test_one_target():
    from spiking_network.stimulation import SinStimulation
    import torch
    targets = 0
    frequencies = 0.01
    amplitudes = 1
    durations = 100
    stim = SinStimulation(
        targets=targets,
        frequencies=frequencies,
        amplitudes=amplitudes,
        durations=durations,
        total_neurons=20,
    )
    torch.testing.assert_close(torch.sum(stim(1)), stim(1)[0])

def test_parameter_dict(regular_stimulation):
    from torch import tensor
    from torch.testing import assert_close
    expected = {'decay': tensor(0.5000), 'strengths': tensor([1., 1., 1.]), 'name': 'RegularStimulation'}
    for key, value in regular_stimulation.parameter_dict.items():
        if isinstance(value, str):
            assert value == expected[key]
        else:
            assert_close(value, expected[key])

def test_target_index_out_of_range():
    from spiking_network.stimulation import SinStimulation
    targets = 20
    frequencies = 0.01
    amplitudes = 1
    durations = 100
    with pytest.raises(ValueError):
        SinStimulation(
            targets=targets,
            frequencies=frequencies,
            amplitudes=amplitudes,
            durations=durations,
            total_neurons=20,
        )

def test_sin_negative_amplitudes():
    from spiking_network.stimulation import SinStimulation
    targets = 0
    frequencies = 0.01
    amplitudes = -1
    durations = 100
    with pytest.raises(ValueError):
        SinStimulation(
            targets=targets,
            frequencies=frequencies,
            amplitudes=amplitudes,
            durations=durations,
            total_neurons=20,
        )

def test_sin_negative_frequencies():
    from spiking_network.stimulation import SinStimulation
    targets = 0
    frequencies = -0.01
    amplitudes = 1
    durations = 100
    with pytest.raises(ValueError):
        SinStimulation(
            targets=targets,
            frequencies=frequencies,
            amplitudes=amplitudes,
            durations=durations,
            total_neurons=20,
        )

def test_sin_high_frequencies():
    from spiking_network.stimulation import SinStimulation
    targets = 0
    frequencies = 100
    amplitudes = 1
    durations = 100
    with pytest.raises(ValueError):
        SinStimulation(
            targets=targets,
            frequencies=frequencies,
            amplitudes=amplitudes,
            durations=durations,
            total_neurons=20,
        )

def test_negative_durations():
    from spiking_network.stimulation import SinStimulation
    targets = 0
    frequencies = 0.01
    amplitudes = 1
    durations = -100
    with pytest.raises(ValueError):
        SinStimulation(
            targets=targets,
            frequencies=frequencies,
            amplitudes=amplitudes,
            durations=durations,
            total_neurons=20,
        )

def test_tuning_with_stimulation(spiking_model, example_data, regular_stimulation):
    from spiking_network.utils import tune
    tunable_parameters = ["threshold", "alpha", "beta"]
    firing_rate = 0.1
    spiking_model.add_stimulation(regular_stimulation)
    tune(spiking_model, example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    assert True

def test_tune_stimulation(spiking_model, example_data, regular_stimulation):
    from spiking_network.utils import tune
    tunable_parameters = ["strengths", "decay"]
    firing_rate = 0.1
    spiking_model.add_stimulation(regular_stimulation)
    tune(spiking_model, example_data, firing_rate, tunable_parameters=tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    assert True

def test_tune_model_and_stimulation(spiking_model, example_data, regular_stimulation):
    from spiking_network.utils import tune
    tunable_model_parameters = ["threshold", "alpha", "beta", "strengths", "decay"]
    firing_rate = 0.1
    spiking_model.add_stimulation(regular_stimulation)
    tune(spiking_model, example_data, firing_rate, tunable_model_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    assert True

def test_simulate_with_stimulation(spiking_model, example_data, regular_stimulation):
    from spiking_network.utils import simulate
    spiking_model.add_stimulation(regular_stimulation)
    simulate(spiking_model, example_data, n_steps=10, verbose=False)
    assert True