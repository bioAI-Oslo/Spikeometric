import pytest
from torch.testing import assert_close
import torch
from torch import mean, std

def test_dataset_size(generated_glorot_data):
    assert len(generated_glorot_data) == 10

def test_normal_params(generated_normal_data):
    for data in generated_normal_data:
        samples = data.W0[data.W0 != 0]
        assert_close(mean(samples), torch.tensor(0.0), atol=5e-2, rtol=5e-2)
        assert_close(std(samples), torch.tensor(1.0), atol=5e-2, rtol=5e-2)

def test_consistent_datasets(saved_glorot_dataset, generated_glorot_data):
    for i in range(len(saved_glorot_dataset)):
        s = saved_glorot_dataset[i]
        g = generated_glorot_data[i]
        assert_close(s.W0, g.W0)
        assert_close(s.edge_index, g.edge_index)
        assert s.num_nodes == g.num_nodes

def test_added_stimulus_masks(saved_glorot_dataset):
    n_neurons = saved_glorot_dataset[0].num_nodes
    stim_masks = [torch.isin(torch.arange(n_neurons), torch.randperm(n_neurons)[:n_neurons//2]) for data in saved_glorot_dataset]
    saved_glorot_dataset.add_stimulus_masks(stim_masks)
    for i in range(len(saved_glorot_dataset)):
        assert_close(saved_glorot_dataset[i].stimulus_mask, stim_masks[i])
        assert saved_glorot_dataset[i].stimulus_mask.dtype == torch.bool
        assert saved_glorot_dataset[i].stimulus_mask.shape == (saved_glorot_dataset[i].num_nodes,)
        assert saved_glorot_dataset[i].stimulus_mask.sum() == saved_glorot_dataset[i].num_nodes // 2

def test_combine_all(saved_glorot_dataset):
    mega_batch = saved_glorot_dataset.combine_all()
    assert mega_batch.num_nodes == sum(data.num_nodes for data in saved_glorot_dataset)
    assert mega_batch.num_edges == sum(data.num_edges for data in saved_glorot_dataset)
    assert mega_batch.W0.shape[0] == sum(data.W0.shape[0] for data in saved_glorot_dataset)

def test_save_from_generator():
    from spikeometric.datasets import NormalGenerator, ConnectivityDataset
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        NormalGenerator(20, mean=0, std=5, glorot=True).save(10, tmpdir)
        dataset = ConnectivityDataset(tmpdir)
        assert len(dataset) == 10
        assert all([data.num_nodes == 20 for data in dataset])

def test_save_and_load_with_stimulus_masks():
    from spikeometric.datasets import NormalGenerator, ConnectivityDataset
    from tempfile import TemporaryDirectory

    stim_masks = [torch.isin(torch.arange(20), torch.randperm(20)[:10]) for _ in range(10)]
    with TemporaryDirectory() as tmpdir:
        NormalGenerator(20, mean=0, std=5, glorot=True).save(10, tmpdir, stimulus_masks=stim_masks)
        dataset = ConnectivityDataset(tmpdir)
        assert len(dataset) == 10
        assert all([data.num_nodes == 20 for data in dataset])
        assert all([torch.equal(data.stimulus_mask, stim_masks[i]) for i, data in enumerate(dataset)])

def test_sparsity_too_low():
    from spikeometric.datasets import NormalGenerator
    with pytest.raises(ValueError):
        NormalGenerator(20, mean=0, std=5, glorot=True, sparsity=0.2)

def test_mexican_hat_dataset(generated_mexican_hat_data):
    assert len(generated_mexican_hat_data) == 10
    assert all([data.num_nodes == 20 for data in generated_mexican_hat_data])
    assert all([not data.has_self_loops() for data in generated_mexican_hat_data])

def test_number_of_neurons_in_dataset(generated_glorot_data):
    assert all([data.num_nodes == 20 for data in generated_glorot_data])

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