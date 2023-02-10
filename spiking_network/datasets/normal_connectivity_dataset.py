import torch
from spiking_network.datasets.connectivity_dataset import ConnectivityDataset
from spiking_network.datasets.distribution_params import DistributionParams

class NormalConnectivityDataset(ConnectivityDataset):
    def __init__(self, n_neurons, n_datasets, distribution_params, seeds=None, transform=None):
        super().__init__(transform=transform)
        self.distribution_params = distribution_params
        if seeds is None:
            seeds = torch.randint(0, 1000000, (n_datasets,)).tolist()

        self.data = self._generate_data(n_neurons, n_datasets, seeds, dist_params=distribution_params)

    def _generate(self, n_neurons: int, seed, dist_params: DistributionParams) -> torch.Tensor:
        """Builds the internal structure of a cluster"""
        rng = torch.Generator().manual_seed(seed)
        if dist_params.name == 'glorot':
            W0 = self._generate_glorot_w0(n_neurons, dist_params.mean, dist_params.std, rng)
        if dist_params.name == 'normal':
            W0 = self._generate_normal_w0(n_neurons, dist_params.mean, dist_params.std, rng)
        W0 = self._dales_law(W0)
        W0 = W0.fill_diagonal_(1)
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
