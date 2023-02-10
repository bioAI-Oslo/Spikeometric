import pytest
import torch

def test_dataset_size(generated_glorot_data):
    assert len(generated_glorot_data) == 10

def test_normal_params(generated_normal_data):
    from torch.testing import assert_close
    from torch import mean, std
    for data in generated_normal_data:
        samples = data.W0[data.W0 != 0]
        assert_close(mean(samples), torch.tensor(0.0), atol=5e-2, rtol=5e-2)
        assert_close(std(samples), torch.tensor(1.0), atol=5e-2, rtol=5e-2)

def test_consistent_datasets(saved_glorot_dataset, generated_glorot_data):
    from torch.testing import assert_close
    for i in range(len(saved_glorot_dataset)):
        s = saved_glorot_dataset[i]
        g = generated_glorot_data[i]
        assert_close(s.W0, g.W0)
        assert_close(s.edge_index, g.edge_index)
        assert s.num_nodes == g.num_nodes

def test_number_of_neurons_in_dataset(generated_glorot_data):
    assert all([data.num_nodes == 20 for data in generated_glorot_data])

def test_self_loops_in_dataset(generated_glorot_data):
    assert all([data.has_self_loops() for data in generated_glorot_data])

def test_uniform_dataset(generated_uniform_data):
    assert len(generated_uniform_data) == 10
    assert all([data.num_nodes == 20 for data in generated_uniform_data])

def test_sparse_glorot_dataset(generated_glorot_data, sparse_glorot_dataset):
    from spikeometric.datasets import NormalGenerator
    assert not torch.equal(sparse_glorot_dataset[0].W0, generated_glorot_data[0].W0)
    assert sparse_glorot_dataset[0].W0.numel() < generated_glorot_data[0].W0.numel()

def test_fails_for_odd_number_of_neurons():
    from spikeometric.datasets import NormalGenerator
    with pytest.raises(ValueError):
        NormalGenerator(21, mean=0, std=5, glorot=True)

def test_numpy_dataset():
    from spikeometric.datasets import ConnectivityDataset
    import shutil
    dataset = ConnectivityDataset(root="tests/test_data/numpy_dataset")
    
    assert len(dataset) == 10
    assert dataset[0].W0.dtype == torch.float32
    assert dataset[0].edge_index.dtype == torch.int64