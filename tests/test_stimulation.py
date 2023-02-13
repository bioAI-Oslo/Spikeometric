import pytest
import torch
from torch.testing import assert_close

@pytest.mark.parametrize("stimulus", [pytest.lazy_fixture('regular_stimulus'), pytest.lazy_fixture('sin_stimulus'), pytest.lazy_fixture('poisson_stimulus')])
def test_zero_after_duration(stimulus):
    stim = stimulus(5000)
    assert_close(stim, torch.tensor(0.))

def test_regular_periods(regular_stimulus, bernoulli_glm):
    bernoulli_glm.add_stimulus(regular_stimulus)
    mask = torch.isin(torch.arange(20), torch.tensor([0, 4, 9]))
    for i in range(0, 1000):
        stim = bernoulli_glm.stimulus_input(i, mask)
        indices = stim.nonzero().squeeze()
        if i % 100 < 10:
            assert 0 in indices
            assert 4 in indices
            assert 9 in indices

def test_regular_no_negative_intervals():
    from spikeometric.stimulus import RegularStimulus
    period = -5
    strength = 1
    tau = 2
    stop = 1000
    with pytest.raises(ValueError):
        RegularStimulus(
            period=period,
            strength=strength,
            tau=tau,
            stop=stop
        )

def test_sin_negative_amplitudes():
    from spikeometric.stimulus import SinStimulus
    period = 100
    amplitude = -1
    duration = 100
    with pytest.raises(ValueError):
        SinStimulus(
            period=period,
            amplitude=amplitude,
            duration=duration,
        )
            
def test_sin_negative_frequencies():
    from spikeometric.stimulus import SinStimulus
    period = -100
    amplitude = 1
    duration = 100
    with pytest.raises(ValueError):
        SinStimulus(
            period=period,
            amplitude=amplitude,
            duration=duration,
        )

def test_negative_durations():
    from spikeometric.stimulus import SinStimulus
    period = 10
    amplitude = 1
    duration = -100
    with pytest.raises(ValueError):
        SinStimulus(
            period=period,
            amplitude=amplitude,
            duration=duration,
        )

def test_poisson_mean_interval(poisson_stimulus):
    mean_rates = poisson_stimulus.stimulus_times.sum() / poisson_stimulus.stimulus_times.max()
    mean_interval = 1 / mean_rates
    pytest.approx(mean_interval, poisson_stimulus.mean_interval)

def test_poisson_non_negative_period():
    from spikeometric.stimulus import PoissonStimulus
    mean_interval = -1
    strength = 1
    duration = 100
    tau = 2
    with pytest.raises(ValueError):
        PoissonStimulus(
            mean_interval=mean_interval,
            strength=strength,
            duration=duration,
            tau=tau,
        )

def test_poisson_non_negative_temporal_scale():
    from spikeometric.stimulus import PoissonStimulus
    mean_interval = 10
    strength = 1
    duration = 100
    tau = -2
    with pytest.raises(ValueError):
        PoissonStimulus(
            mean_interval=mean_interval,
            strength=strength,
            duration=duration,
            tau=tau,
        )

def test_manual_stimulus(example_data, bernoulli_glm):
    import numpy as np
    initial_state = torch.zeros((example_data.num_nodes, bernoulli_glm.T))
    initial_state[:, -1] = torch.randint(0, 2, (example_data.num_nodes,), generator=torch.Generator().manual_seed(14071789))

    connectivity_filter, edge_index = bernoulli_glm.connectivity_filter(example_data.W0, example_data.edge_index)
    input_without_stimulus = bernoulli_glm.synaptic_input(edge_index, connectivity_filter, state=initial_state, t=0)
    
    n_neurons = example_data.num_nodes
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    stim_mask = torch.isin(torch.arange(n_neurons), targets)

    func = lambda t: np.cos(t)
    bernoulli_glm.add_stimulus(func)
    input_with_stimulus = bernoulli_glm.input(edge_index, connectivity_filter, state=initial_state, t=0, stimulus_mask=stim_mask)
    assert any(input_with_stimulus != input_without_stimulus)

def test_tuning_with_stimulus(bernoulli_glm, example_data, regular_stimulus):
    tunable_parameters = ["theta", "alpha", "beta"]
    firing_rate = 0.1
    bernoulli_glm.add_stimulus(regular_stimulus)
    example_data.targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    bernoulli_glm.tune(example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    assert True

@pytest.mark.parametrize("stimulus", [pytest.lazy_fixture("regular_stimulus"), pytest.lazy_fixture("poisson_stimulus")])
def test_tune_stimulus(bernoulli_glm, example_data, stimulus):
    tunable_parameters = ["stimulus.strength"]
    firing_rate = 0.1
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    example_data.stimulus_mask = torch.isin(torch.arange(example_data.num_nodes), targets)
    bernoulli_glm.add_stimulus(stimulus)
    bernoulli_glm.tune(example_data, firing_rate, tunable_parameters=tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    assert True

def test_tune_model_and_stimulus(bernoulli_glm, example_data, regular_stimulus):
    tunable_model_parameters = ["theta", "alpha", "beta", "stimulus.strength"]
    firing_rate = 0.1
    bernoulli_glm.add_stimulus(regular_stimulus)
    example_data.stimulus_mask = torch.isin(torch.arange(example_data.num_nodes), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    bernoulli_glm.tune(example_data, firing_rate, tunable_model_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    assert True

def test_simulate_with_stimulus(bernoulli_glm, example_data, regular_stimulus):
    bernoulli_glm.add_stimulus(regular_stimulus)
    example_data.stimulus_mask = torch.isin(torch.arange(example_data.num_nodes), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    bernoulli_glm.simulate(example_data, n_steps=10, verbose=False)
    assert True