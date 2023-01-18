from spiking_network.utils import simulate
from torch.testing import assert_close
import pytest

def test_no_grad(glm_model, example_data):
    X = simulate(glm_model, example_data, n_steps=10, verbose=False)
    assert not X.requires_grad

def test_consistent_output_after_one_step(expected_state_after_one_step, glm_model, example_data):
    X = simulate(glm_model, example_data, n_steps=1, verbose=False)
    assert_close(X.squeeze(), expected_state_after_one_step)

def test_consistent_output_after_ten_steps(expected_output_after_ten_steps, glm_model, example_data):
    X = simulate(glm_model, example_data, n_steps=10, verbose=False)
    assert_close(X, expected_output_after_ten_steps)

def test_simulation_statistics(glm_model, saved_glorot_dataset):
    from spiking_network.utils import calculate_firing_rate
    n_steps = 1000
    expected_firing_rate = 0.0072
    for example_data in saved_glorot_dataset:
        X = simulate(glm_model, example_data, n_steps=n_steps, verbose=False)
        pytest.approx(calculate_firing_rate(X), expected_firing_rate)

def test_save_load_output(expected_output_after_ten_steps, glm_model, example_data):
    from pathlib import Path
    from spiking_network.utils import save_data, load_data
    from torch import sparse_coo_tensor

    # Save data
    path = Path("tests/test_data")
    save_data(expected_output_after_ten_steps, glm_model, [example_data], seed=14071789, data_path=path)

    # Load data
    expected_X, expected_W0 = load_data(path / "0.npz")
    dense_W0 = sparse_coo_tensor(example_data.edge_index, example_data.W0, (example_data.num_nodes, example_data.num_nodes)).to_dense()

    # Check data
    assert_close(expected_X, expected_output_after_ten_steps)
    assert_close(expected_W0, dense_W0)

    # Clean up
    (path / "0.npz").unlink()

def test_time_to_simulate_100_steps(glm_model, example_data):
    from benchmarking.timing import time_model
    expected_time = 0.011739735399532946
    t = time_model(model=glm_model, data=example_data, n_steps=100, N=10)
    assert t <= expected_time + 0.01

def test_uniform_simulation(lnp_model, example_uniform_data):
    from spiking_network.utils import calculate_firing_rate
    from torch import tensor
    X = simulate(lnp_model, example_uniform_data, n_steps=1000, verbose=False)
    assert_close(calculate_firing_rate(X), tensor(0.006), atol=0.001, rtol=0.1)