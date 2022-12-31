import pytest
import torch

@pytest.mark.parametrize("dataset", [pytest.lazy_fixture('generated_dataset'), pytest.lazy_fixture('loaded_dataset')])
def test_dataset_size(dataset):
    assert len(dataset) == 10

def test_normal_params(generated_normal_dataset):
    from torch.testing import assert_close
    from torch import mean, std
    for data in generated_normal_dataset:
        samples = data.W0[data.W0 != 0]
        assert_close(mean(samples), torch.tensor(0.0), atol=5e-2, rtol=5e-2)
        assert_close(std(samples), torch.tensor(1.0), atol=5e-2, rtol=5e-2)

def test_consistent_datasets(saved_dataset, generated_dataset):
    from torch.testing import assert_close
    for s, g in zip(saved_dataset, generated_dataset):
        assert_close(s.W0, g.W0)
        assert_close(s.edge_index, g.edge_index)
        assert s.num_nodes == g.num_nodes

def test_fails_for_non_square_matrix():
    from torch.testing import make_tensor
    from spiking_network.datasets import ConnectivityDataset
    with pytest.raises(ValueError):
        t = make_tensor((10, 20), device='cpu', dtype=torch.float32)
        dataset = ConnectivityDataset([t])

def test_numpy_array():
    import numpy as np
    from spiking_network.datasets import ConnectivityDataset
    t = np.random.rand(10, 10)
    dataset = ConnectivityDataset([t])
    assert isinstance(dataset[0].W0, torch.Tensor)

def test_geometric_data(example_data):
    from spiking_network.datasets import ConnectivityDataset
    dataset = ConnectivityDataset([example_data])
    assert isinstance(dataset[0].W0, torch.Tensor)
    assert isinstance(dataset[0].edge_index, torch.Tensor)

def test_fails_for_nonsense_data():
    from spiking_network.datasets import ConnectivityDataset
    with pytest.raises(ValueError):
        dataset = ConnectivityDataset([1])

def test_different_data(generated_dataset):
    for i in range(1, len(generated_dataset)):
        assert not torch.equal(generated_dataset[i].W0, generated_dataset[i-1].W0)

def test_fails_for_odd_number_of_neurons():
    from spiking_network.datasets import W0Dataset, GlorotParams
    with pytest.raises(ValueError):
        W0Dataset(21, 10, GlorotParams(0, 5))

def test_fails_for_negative_number_of_sims():
    from spiking_network.datasets import W0Dataset, GlorotParams
    with pytest.raises(ValueError):
        W0Dataset(10, -1, GlorotParams(0, 5))

def test_to_dense(loaded_dataset):
    from torch import sparse_coo_tensor
    from torch.testing import assert_close
    dense_data = loaded_dataset.to_dense()[0]
    w0 = loaded_dataset[0].W0
    edge_index = loaded_dataset[0].edge_index
    sparse_data = sparse_coo_tensor(edge_index, w0, dense_data.shape)
    assert_close(dense_data, sparse_data.to_dense())
    
@pytest.mark.parametrize("dataset", [pytest.lazy_fixture('generated_dataset'), pytest.lazy_fixture('loaded_dataset')])
def test_number_of_neurons_in_dataset(dataset):
    assert all([data.num_nodes == 20 for data in dataset])

@pytest.mark.parametrize("dataset", [pytest.lazy_fixture('generated_dataset'), pytest.lazy_fixture('loaded_dataset')])
def test_self_loops_in_dataset(dataset):
    assert all([data.has_self_loops() for data in dataset])

def test_herman_dataset():
    from spiking_network.datasets import HermanDataset
    dataset = HermanDataset(20, 10)
    assert len(dataset) == 10
    assert all([data.num_nodes == 20 for data in dataset])
    assert all([data.has_self_loops() for data in dataset])

def test_sparse_dataset():
    from spiking_network.datasets import W0Dataset, GlorotParams
    sparse_dataset = W0Dataset(20, 1, GlorotParams(0, 5), sparsity=0.5)
    dataset = W0Dataset(20, 1, GlorotParams(0, 5))
    assert not torch.equal(sparse_dataset[0].W0, dataset[0].W0)
    assert sparse_dataset[0].W0.numel() < dataset[0].W0.numel()