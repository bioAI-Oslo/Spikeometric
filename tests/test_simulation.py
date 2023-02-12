from torch.testing import assert_close
import pytest

def test_no_grad(bernoulli_glm, example_data):
    X = bernoulli_glm.simulate(example_data, n_steps=1, verbose=False)
    assert not bernoulli_glm.alpha.requires_grad

def test_consistent_output_after_ten_steps(expected_output_after_ten_steps, bernoulli_glm, example_data):
    X = bernoulli_glm.simulate(example_data, n_steps=10, verbose=False, equilibration_steps=0)
    assert_close(X, expected_output_after_ten_steps)

def test_simulation_statistics(bernoulli_glm, saved_glorot_dataset):
    n_steps = 1000
    expected_firing_rate = 7.2
    for example_data in saved_glorot_dataset:
        X = bernoulli_glm.simulate(example_data, n_steps=n_steps, verbose=False)
        fr = (X.float().mean() / bernoulli_glm.dt) * 1000
        pytest.approx(fr, expected_firing_rate)

def test_uniform_simulation(threshold_sam, generated_uniform_data):
    from torch import tensor
    example_uniform_data = generated_uniform_data[0]
    X = threshold_sam.simulate(example_uniform_data, n_steps=1000, verbose=False)
    fr = (X.float().mean() / threshold_sam.dt) * 1000
    assert_close(fr, tensor(33.6364), atol=0.001, rtol=0.1)