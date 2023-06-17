import pytest
import torch
from torch.testing import assert_close

@pytest.mark.parametrize("stimulus", [pytest.lazy_fixture('regular_stimulus'), pytest.lazy_fixture('sin_stimulus'), pytest.lazy_fixture('poisson_stimulus')])
def test_zero_after_duration(stimulus):
    stim = stimulus(5000)
    n_neurons = stim.shape[0]
    assert_close(stim, torch.zeros(n_neurons))

def test_regular_periods(regular_stimulus, bernoulli_glm):
    bernoulli_glm.add_stimulus(regular_stimulus)
    for i in range(0, 1000):
        stim = bernoulli_glm.stimulus_input(i)
        indices = stim.nonzero().squeeze()
        if i % 100 < 10:
            assert 0 in indices
            assert 2 in indices
            assert 4 in indices
            assert 6 in indices
            assert 8 in indices

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
            stop=stop,
            stimulus_masks=torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)
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
            stimulus_masks=torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)
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
            stimulus_masks=torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)
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
            stimulus_masks=torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)
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
            stimulus_masks=torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)
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
            stimulus_masks=torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)
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

    func = lambda t: np.cos(t)*stim_mask
    bernoulli_glm.add_stimulus(func)
    input_with_stimulus = bernoulli_glm.input(edge_index, connectivity_filter, state=initial_state, t=0)
    assert any(input_with_stimulus != input_without_stimulus)

@pytest.mark.parametrize("stimulus", [pytest.lazy_fixture("regular_stimulus"), pytest.lazy_fixture("poisson_stimulus"), pytest.lazy_fixture("loaded_stimulus")])
def test_tuning_with_stimulus(bernoulli_glm, example_data, stimulus):
    tunable_parameters = ["theta", "alpha", "beta"]
    firing_rate = 0.1
    bernoulli_glm.add_stimulus(stimulus)
    bernoulli_glm.tune(example_data, firing_rate, tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    assert True

@pytest.mark.parametrize("stimulus", [pytest.lazy_fixture("regular_stimulus"), pytest.lazy_fixture("poisson_stimulus")])
def test_tune_stimulus(bernoulli_glm, example_data, stimulus):
    tunable_parameters = ["stimulus.strength"]
    firing_rate = 0.1
    bernoulli_glm.add_stimulus(stimulus)
    bernoulli_glm.tune(example_data, firing_rate, tunable_parameters=tunable_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    assert True

@pytest.mark.parametrize("stimulus", [pytest.lazy_fixture("regular_stimulus"), pytest.lazy_fixture("poisson_stimulus")])
def test_tune_model_and_stimulus(bernoulli_glm, example_data, stimulus):
    tunable_model_parameters = ["theta", "alpha", "beta", "stimulus.strength"]
    firing_rate = 0.1
    bernoulli_glm.add_stimulus(stimulus)
    bernoulli_glm.tune(example_data, firing_rate, tunable_model_parameters, n_steps=10, n_epochs=1, lr=0.1, verbose=False)
    assert True

@pytest.mark.parametrize("stimulus", [pytest.lazy_fixture("regular_stimulus"), pytest.lazy_fixture("poisson_stimulus"), pytest.lazy_fixture("loaded_stimulus")])
def test_simulate_with_stimulus(bernoulli_glm, example_data, stimulus):
    bernoulli_glm.add_stimulus(stimulus)
    bernoulli_glm.simulate(example_data, n_steps=10, verbose=False)
    assert True

def test_stimulus_masks_are_batched():
    from spikeometric.stimulus import RegularStimulus, PoissonStimulus, SinStimulus
    stim_masks = [torch.isin(torch.arange(10), torch.randperm(10)[:5]) for _ in range(5)]
    stimuli = [
        RegularStimulus(1, 100, 10, 1000, stim_masks, 2),
        PoissonStimulus(1, 10, 1000, stim_masks, 2),
        SinStimulus(1, 10, 1000, stim_masks, 2)
    ]
    for stimulus in stimuli:
        assert stimulus(0).shape == (20,)
        assert stimulus(1000).shape == (20,)
        stimulus.set_batch(2)
        assert stimulus(0).shape == (10,)
        assert stimulus(1000).shape == (10,)

def test_stimulus_batching_fails_if_batch_size_is_too_large():
    from spikeometric.stimulus import RegularStimulus
    stim_mask = torch.isin(torch.arange(10), torch.randperm(10)[:5])
    with pytest.raises(ValueError):
        stimulus = RegularStimulus(1, 100, 10, 1000, stim_mask, 2)

def test_stimulus_batching_works_for_tensor_input_of_stim_masks():
    from spikeometric.stimulus import RegularStimulus, PoissonStimulus, SinStimulus
    stim_masks = torch.rand(5, 10) > 0.5
    stimuli = [
        RegularStimulus(1, 100, 10, 1000, stim_masks, 2),
        PoissonStimulus(1, 10, 1000, stim_masks, 2),
        SinStimulus(1, 10, 1000, stim_masks, 2)
    ]
    for stimulus in stimuli:
        assert stimulus(0).shape == (20,)
        assert stimulus(1000).shape == (20,)
        stimulus.set_batch(2)
        assert stimulus(0).shape == (10,)
        assert stimulus(1000).shape == (10,)

def test_regular_stimulus_vectorizes(regular_stimulus):
    t = torch.arange(0, 1000)
    assert regular_stimulus(t).shape == (20, 1000)

def test_poisson_stimulus_vectorizes(poisson_stimulus):
    t = torch.arange(0, 1000)
    assert poisson_stimulus(t).shape == (20, 1000)

def test_sin_stimulus_vectorizes(sin_stimulus):
    t = torch.arange(0, 1000)
    assert sin_stimulus(t).shape == (20, 1000)

def test_loaded_stimulus_is_batched():
    from spikeometric.stimulus import LoadedStimulus
    stimulus = LoadedStimulus("tests/test_data/stim_plan_4_networks.pt", batch_size=2)
    n_steps = stimulus.n_steps.item()
    assert stimulus(0).shape == (40,)
    assert stimulus(n_steps).shape == (40,)

def test_loaded_stimulus_can_batch_non_uneven_number_of_batches():
    from spikeometric.stimulus import LoadedStimulus
    stimulus = LoadedStimulus("tests/test_data/stim_plan_4_networks.pt", batch_size=3)
    n_steps = stimulus.n_steps.item()
    assert stimulus(0).shape == (60,)
    assert stimulus(n_steps+1).shape == (60,)
    stimulus.next_batch()
    assert stimulus(0).shape == (20,)
    assert stimulus(n_steps+1).shape == (20,)

def test_loaded_stimulus_affects_model_output(bernoulli_glm, example_data, loaded_stimulus):
    initial_spikes = bernoulli_glm.simulate(example_data, n_steps=100, verbose=False)
    bernoulli_glm.add_stimulus(loaded_stimulus)

    spikes_with_stimulus = bernoulli_glm.simulate(example_data, n_steps=100, verbose=False)
    assert spikes_with_stimulus.sum() > initial_spikes.sum()

def test_loaded_stimulus_matches_original_stimulus(loaded_stimulus):
    original_stimulus = torch.load("tests/test_data/stim_plan.pt")
    for i in range(loaded_stimulus.n_steps):
        assert torch.allclose(loaded_stimulus(i), original_stimulus[:, i])

    n_steps = loaded_stimulus.n_steps.item()
    assert torch.allclose(loaded_stimulus(n_steps+1), torch.zeros(20))

def test_loaded_stimulus_is_zero_before_start_of_stimulus(loaded_stimulus):
    assert torch.allclose(loaded_stimulus(-1), torch.zeros(20))

def test_loaded_stimulus_cycles_through_batches():
    from spikeometric.stimulus import LoadedStimulus
    stimulus = LoadedStimulus("tests/test_data/stim_plan_4_networks.pt", batch_size=2)
    n_steps = stimulus.n_steps
    first_batch = stimulus(torch.arange(0, n_steps))
    stimulus.next_batch()
    second_batch = stimulus(torch.arange(0, n_steps))
    stimulus.next_batch()
    assert torch.allclose(first_batch, stimulus(torch.arange(0, n_steps)))
    stimulus.next_batch()
    assert torch.allclose(second_batch, stimulus(torch.arange(0, n_steps)))

def test_loaded_stimulus_fails_if_batch_size_greater_than_n_networks():
    from spikeometric.stimulus import LoadedStimulus
    with pytest.raises(ValueError):
        stimulus = LoadedStimulus("tests/test_data/stim_plan_4_networks.pt", batch_size=5)

def test_loaded_stimulus_fails_if_idx_out_of_range():
    from spikeometric.stimulus import LoadedStimulus
    stimulus = LoadedStimulus("tests/test_data/stim_plan_4_networks.pt", batch_size=2)
    with pytest.raises(ValueError):
        stimulus.set_batch(2)

def test_loaded_stimulus_resets():
    from spikeometric.stimulus import LoadedStimulus
    stimulus = LoadedStimulus("tests/test_data/stim_plan_4_networks.pt", batch_size=1)
    n_steps = stimulus.n_steps
    first_batch = stimulus(torch.arange(0, n_steps))
    stimulus.set_batch(3)
    stimulus.reset()
    assert torch.allclose(first_batch, stimulus(torch.arange(0, n_steps)))

