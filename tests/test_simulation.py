from torch.testing import assert_close
import pytest

def test_no_grad(bernoulli_glm, example_data):
    X = bernoulli_glm.simulate(example_data, n_steps=1, verbose=False)
    assert not bernoulli_glm.alpha.requires_grad

def test_consistent_output_after_ten_steps(expected_output_after_ten_steps, bernoulli_glm, example_data):
    X = bernoulli_glm.simulate(example_data, n_steps=10, verbose=False, equilibration_steps=0)
    assert_close(X, expected_output_after_ten_steps)

def test_simulation_statistics(bernoulli_glm, saved_glorot_dataset):
    from spiking_network.utils import calculate_firing_rate
    n_steps = 1000
    expected_firing_rate = 7.2
    for example_data in saved_glorot_dataset:
        X = bernoulli_glm.simulate(example_data, n_steps=n_steps, verbose=False)
        pytest.approx(calculate_firing_rate(X, bernoulli_glm.dt), expected_firing_rate)

def test_save_load_output(expected_output_after_ten_steps, bernoulli_glm, example_data):
    from pathlib import Path
    from spiking_network.utils import save_data, load_data
    from torch import sparse_coo_tensor

    # Save data
    path = Path("tests/test_data")
    save_data(expected_output_after_ten_steps, bernoulli_glm, [example_data], seed=14071789, data_path=path)

    # Load data
    expected_X, expected_W0 = load_data(path / "0.npz")
    dense_W0 = sparse_coo_tensor(example_data.edge_index, example_data.W0, (example_data.num_nodes, example_data.num_nodes)).to_dense()

    # Check data
    assert_close(expected_X, expected_output_after_ten_steps)
    assert_close(expected_W0, dense_W0)

    # Clean up
    (path / "0.npz").unlink()

def test_time_to_simulate_100_steps(bernoulli_glm, example_data):
    from benchmarking.timing import time_model
    expected_time = 4.950151900120545e-05
    t = time_model(model=bernoulli_glm, data=example_data, n_steps=1000, N=10)
    assert t <= expected_time + 0.0001

def test_uniform_simulation(threshold_sam, generated_uniform_data):
    from spiking_network.utils import calculate_firing_rate
    from torch import tensor
    example_uniform_data = generated_uniform_data[0]
    X = threshold_sam.simulate(example_uniform_data, n_steps=1000, verbose=False)
    assert_close(calculate_firing_rate(X, threshold_sam.dt), tensor(27.), atol=0.001, rtol=0.1)