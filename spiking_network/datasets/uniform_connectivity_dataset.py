import numpy as np
import torch
from spiking_network.datasets.connectivity_dataset import ConnectivityDataset

class UniformConnectivityDataset(ConnectivityDataset):
    UNIFORM_LOWEST = -0.002289225919299652
    def __init__(self, n_neurons, n_datasets, sparsity=0.9, seed=0):
        self.sparsity = sparsity
        self.data = self._generate_data(n_neurons, n_datasets, seed)

    def _generate(self, n_neurons, seed):
        rng = np.random.default_rng(seed)
        mat = rng.uniform(self.UNIFORM_LOWEST, 0, size=(n_neurons, n_neurons))
        mat[rng.random((n_neurons, n_neurons)) < self.sparsity] = 0
        w0 = torch.tensor(mat, dtype=torch.float32)
        return w0
