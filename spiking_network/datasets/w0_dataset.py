import torch
from spiking_network.datasets.distribution_params import DistributionParams
from spiking_network.datasets.connectivity_dataset import ConnectivityDataset
from dataclasses import dataclass

class W0Dataset(ConnectivityDataset):
    def __init__(self, n_neurons: int, n_datasets: int, distribution_params: DistributionParams, sparsity=0, seed=None):
        if n_neurons % 2 != 0 or n_neurons < 2:
            raise ValueError("n_neurons must be positive and even to have as many excitatory as inhibitory neurons")
        self.n_neurons = n_neurons

        if n_datasets < 1:
            raise ValueError("n_datasets must be positive")
        self.n_datasets = n_datasets

        self.distribution_params = distribution_params

        if seed:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)
        else:
            self._rng = torch.Generator()

        w0_list = []
        for i in range(n_datasets):
            W0 = self._generate(n_neurons, distribution_params, sparsity)
            w0_list.append(W0)

        # Add the data to the dataset
        super().__init__(w0_list)

    def _generate(self, n_neurons: int, dist_params: DistributionParams, sparsity: float) -> torch.Tensor:
        """Builds the internal structure of a cluster"""
        if dist_params.name == 'glorot':
            W0 = self._generate_glorot_w0(n_neurons, dist_params.mean, dist_params.std)
        if dist_params.name == 'normal':
            W0 = self._generate_normal_w0(n_neurons, dist_params.mean, dist_params.std)
        W0 = self._dales_law(W0)
        W0[torch.eye(n_neurons, dtype=torch.bool)] = 0
        W0[W0 != 0] = W0[W0 != 0] * (torch.rand_like(W0[W0 != 0]) > sparsity)
        return W0

    def _dales_law(self, W0: torch.Tensor) -> torch.Tensor:
        """Applies Dale's law to the connectivity matrix W0"""
        W0 = torch.concat((W0 * (W0 > 0), W0 * (W0 < 0)), 0)
        return W0

    def _generate_normal_w0(self, n_neurons: int, mean: float, std: float) -> torch.Tensor:
        """Generates a normal n_neurons/2*n_neurons/2 matrux from a normal distribution"""
        half_n_neurons = n_neurons // 2
        W0 = torch.normal(mean, std, (half_n_neurons, n_neurons), generator=self._rng)
        return W0
    
    def _generate_glorot_w0(self, n_neurons: int, mean: float, std: float) -> torch.Tensor:
        """Generates a normal n_neurons/2*n_neurons/2 matrux from a normal distribution"""
        normal_W0 = self._generate_normal_w0(n_neurons, mean, std)
        glorot_W0 = normal_W0 / torch.sqrt(torch.tensor(n_neurons, dtype=torch.float32))
        return glorot_W0


