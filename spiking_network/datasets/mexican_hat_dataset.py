import numpy as np
import torch
from spiking_network.datasets.connectivity_dataset import ConnectivityDataset
from pathlib import Path

class MexicanHatDataset(ConnectivityDataset):
    MEXICAN_HAT_LOWEST = -0.002289225919299652
    def __init__(self, n_neurons, n_examples, root, emptiness=0.9, seed=0, transform=None, pre_transform=None):
        self.emptiness = emptiness

        path = Path(root) / "raw"
        path.mkdir(parents=True, exist_ok=True)
        for i in range(n_examples):
            W0 = self._generate(n_neurons, seed)
            torch.save(W0, path / f"example_{i}.pt")

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _generate(self, n_neurons, seed):
        rng = np.random.default_rng(seed)
        mat = rng.uniform(self.MEXICAN_HAT_LOWEST, 0, size=(n_neurons, n_neurons))
        mat[rng.random((n_neurons, n_neurons)) < self.emptiness] = 0
        w0 = torch.tensor(mat, dtype=torch.float32)
        return w0
