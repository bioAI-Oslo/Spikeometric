import numpy as np
import torch
from spiking_network.datasets.connectivity_dataset import ConnectivityDataset
from pathlib import Path

class UniformConnectivityDataset(ConnectivityDataset):
    UNIFORM_LOWEST = -0.002289225919299652
    def __init__(self, n_neurons, n_examples, root, sparsity=0.9, rng=None, transform=None, pre_transform=None):
        self.sparsity = sparsity

        path = Path(root) / "raw"
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            self._rng = rng if rng is not None else torch.Generator()
            for i in range(n_examples):
                W0 = self._generate(n_neurons)
                torch.save(W0, path / f"example_{i}.pt")

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _generate(self, n_neurons):
        w0 = torch.rand(size=(n_neurons, n_neurons), generator=self._rng)*self.UNIFORM_LOWEST
        w0[torch.rand((n_neurons, n_neurons)) < self.sparsity] = 0
        return w0
