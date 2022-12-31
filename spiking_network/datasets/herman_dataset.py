import numpy as np
import torch
from spiking_network.datasets.connectivity_dataset import ConnectivityDataset

class HermanDataset(ConnectivityDataset):
    MEXICAN_HAT_LOWEST = -0.002289225919299652
    def __init__(self, n_neurons, n_datasets, emptiness=0.9, seed=0):
        self.emptiness = emptiness

        w0_list = []
        for _ in range(n_datasets):
            w0_list.append(self._generate(n_neurons, seed))
        
        super().__init__(w0_list)

    def _generate(self, n_neurons, seed):
        rng = np.random.default_rng(seed)
        mat = rng.uniform(self.MEXICAN_HAT_LOWEST, 0, size=(n_neurons, n_neurons))
        mat[rng.random((n_neurons, n_neurons)) < self.emptiness] = 0
        w0 = torch.tensor(mat, dtype=torch.float32)
        return w0

    def name(self):
        return 'herman'
