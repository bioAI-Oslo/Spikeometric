import pytest
import torch

def test_dataset_size(generated_glorot_dataset):
    assert len(generated_glorot_dataset) == 10

def test_normal_params(generated_normal_dataset):
    from torch.testing import assert_close
    from torch import mean, std
    for data in generated_normal_dataset:
        samples = data.W0[data.W0 != 0]
        assert_close(mean(samples), torch.tensor(0.0), atol=5e-2, rtol=5e-2)
        assert_close(std(samples), torch.tensor(1.0), atol=5e-2, rtol=5e-2)

def test_consistent_datasets(saved_glorot_dataset, generated_glorot_dataset):
    from torch.testing import assert_close
    for i in range(len(saved_glorot_dataset)):
        s = saved_glorot_dataset[i]
        g = generated_glorot_dataset[i]
        assert_close(s.W0, g.W0)
        assert_close(s.edge_index, g.edge_index)
        assert s.num_nodes == g.num_nodes

def test_number_of_neurons_in_dataset(generated_glorot_dataset):
    assert all([data.num_nodes == 20 for data in generated_glorot_dataset])

def test_self_loops_in_dataset(generated_glorot_dataset):
    assert all([data.has_self_loops() for data in generated_glorot_dataset])

def test_uniform_dataset(generated_uniform_dataset):
    assert len(generated_uniform_dataset) == 10
    assert all([data.num_nodes == 20 for data in generated_uniform_dataset])
    assert all([data.has_self_loops() for data in generated_uniform_dataset])

def test_sparse_glorot_dataset(generated_glorot_dataset, sparse_glorot_dataset):
    from spiking_network.datasets import NormalConnectivityDataset, GlorotParams
    assert not torch.equal(sparse_glorot_dataset[0].W0, generated_glorot_dataset[0].W0)
    assert sparse_glorot_dataset[0].W0.numel() < generated_glorot_dataset[0].W0.numel()

def test_fails_for_odd_number_of_neurons():
    from spiking_network.datasets import NormalConnectivityDataset, GlorotParams
    with pytest.raises(ValueError):
        NormalConnectivityDataset(21, 10, GlorotParams(0, 5), root="")

def test_fails_for_negative_number_of_sims():
    from spiking_network.datasets import NormalConnectivityDataset, GlorotParams
    with pytest.raises(ValueError):
        NormalConnectivityDataset(10, -1, GlorotParams(0, 5), root="")

def test_numpy_dataset():
    from spiking_network.datasets import ConnectivityDataset
    import shutil
    dataset = ConnectivityDataset(root="tests/test_data/numpy_dataset")
    shutil.rmtree("tests/test_data/numpy_dataset/processed") # remove the processed folder so that the dataset is reprocessed

    assert len(dataset) == 10
    assert dataset[0].W0.dtype == torch.float32
    assert dataset[0].edge_index.dtype == torch.int64


def test_to_dense(saved_glorot_dataset):
    from torch import sparse_coo_tensor
    from torch.testing import assert_close
    dense_data = saved_glorot_dataset.to_dense()[0]
    w0 = saved_glorot_dataset[0].W0
    edge_index = saved_glorot_dataset[0].edge_index
    sparse_data = sparse_coo_tensor(edge_index, w0, dense_data.shape)
    assert_close(dense_data, sparse_data.to_dense())

def test_generate_examples(saved_glorot_dataset):
    from spiking_network.datasets import NormalConnectivityDataset, GlorotParams
    from torch.testing import assert_close
    data = NormalConnectivityDataset.generate_examples(20, 10, GlorotParams(0, 5), seed=14071789)
    for i in range(len(data)):
        assert_close(data[i].W0, saved_glorot_dataset[i].W0)
        assert_close(data[i].edge_index, saved_glorot_dataset[i].edge_index)