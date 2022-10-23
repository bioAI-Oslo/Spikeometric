import torch
import numpy as np
from torch_geometric.data import Data
from spiking_network.w0_generators.w0_generator import DistributionParams

class ConnectivityDataset:
    @classmethod
    def from_list(cls, w0_list):
        dataset = cls()
        dataset.data = []
        for w0 in w0_list:
            n_neurons = w0.shape[0]
            edge_index = w0.nonzero().t()
            w0 = w0[edge_index[0], edge_index[1]]
            dataset.data.append(Data(W0=w0, edge_index=edge_index, num_nodes=n_neurons))
        return dataset

    @property
    def w0_list(self):
        return [data.W0 for data in self.data]

    @property
    def edge_index_list(self):
        return [data.edge_index for data in self.data]

    def _generate_data(self, n_neurons, n_datasets, seed=0, **kwargs):
        datasets = []
        rng = torch.Generator()
        for i in range(n_datasets):
            W0 = self._generate(n_neurons, seed, **kwargs)
            edge_index = W0.nonzero().t()
            W0 = W0[edge_index[0], edge_index[1]]
            datasets.append(Data(W0=W0, edge_index=edge_index, num_nodes=n_neurons))
            seed += 1
        return datasets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class W0Dataset(ConnectivityDataset):
    def __init__(self, n_neurons, n_datasets, distribution_params, seed=0):
        self.distribution_params = distribution_params
        self.data = self._generate_data(n_neurons, n_datasets, seed, dist_params=distribution_params)

    def _generate(self, n_neurons: int, seed, dist_params: DistributionParams) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds the internal structure of a cluster"""
        rng = torch.Generator().manual_seed(seed)
        if dist_params.name == 'glorot':
            W0 = self._generate_glorot_w0(n_neurons, dist_params.mean, dist_params.std, rng)
        if dist_params.name == 'normal':
            W0 = self._generate_normal_w0(n_neurons, dist_params.mean, dist_params.std, rng)
        W0 = self._dales_law(W0)
        W0 = W0.fill_diagonal_(1)
        return W0

    def _dales_law(self, W0: torch.Tensor) -> torch.Tensor:
        """Applies Dale's law to the connectivity matrix W0"""
        W0 = torch.concat((W0 * (W0 > 0), W0 * (W0 < 0)), 0)
        return W0

    def _generate_normal_w0(self, n_neurons: int, mean: float, std: float, rng: torch.Generator) -> torch.Tensor:
        """Generates a normal n_neurons/2*n_neurons/2 matrux from a normal distribution"""
        half_n_neurons = n_neurons // 2
        W0 = torch.normal(mean, std, (half_n_neurons, n_neurons), generator=rng)
        return W0
    
    def _generate_glorot_w0(self, n_neurons: int, mean: float, std: float, rng: torch.Generator) -> torch.Tensor:
        """Generates a normal n_neurons/2*n_neurons/2 matrux from a normal distribution"""
        normal_W0 = self._generate_normal_w0(n_neurons, mean, std, rng)
        glorot_W0 = normal_W0 / torch.sqrt(torch.tensor(n_neurons, dtype=torch.float32))
        return glorot_W0

class HermanDataset(ConnectivityDataset):
    MEXICAN_HAT_LOWEST = -0.002289225919299652
    def __init__(self, n_neurons, n_datasets, seed=0):
        self.data = self._generate_data(n_neurons, n_datasets, seed)
    
    def _generate(self, n_neurons, seed):
        rng = np.random.default_rng(seed)
        mat = rng.uniform(self.MEXICAN_HAT_LOWEST, 0, size=(n_neurons, n_neurons))
        mat[rng.random((n_neurons, n_neurons)) < 0.9] = 0
        w0 = torch.tensor(mat, dtype=torch.float32)
        return w0
    

