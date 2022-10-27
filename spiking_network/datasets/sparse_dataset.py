import torch
from spiking_network.datasets.w0_dataset import W0Dataset, DistributionParams

class SparseW0Dataset(W0Dataset):
    def __init__(self, n_neurons, n_datasets, distribution_params, emptiness, seed=0):
        self.emptiness = emptiness
        super().__init__(n_neurons, n_datasets, distribution_params, seed)

    def _generate(self, n_neurons: int, seed, dist_params: DistributionParams) -> torch.Tensor:
        """Builds the internal structure of a cluster"""
        W0 = super()._generate(n_neurons, seed, dist_params)
        W0 = W0 * (torch.rand_like(W0) > self.emptiness)
        return W0
