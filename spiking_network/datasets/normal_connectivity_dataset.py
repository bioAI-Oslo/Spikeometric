import torch
from spiking_network.datasets.distribution_params import DistributionParams
from spiking_network.datasets.connectivity_dataset import ConnectivityDataset
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops

class NormalConnectivityDataset(ConnectivityDataset):
    def __init__(self, n_neurons: int, n_examples: int, distribution_params: DistributionParams, root, sparsity=0, rng=None, transform=None, pre_transform=None):
        if n_neurons % 2 != 0 or n_neurons < 2:
            raise ValueError("n_neurons must be positive and even to have as many excitatory as inhibitory neurons")
        self.n_neurons = n_neurons

        if n_examples < 1:
            raise ValueError("n_examples must be positive")

        path = Path(root) / "raw"
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            self.distribution_params = distribution_params
            self._rng = rng if rng is not None else torch.Generator()
            
            for i in range(n_examples):
                W0 = self._generate(n_neurons, distribution_params, sparsity, self._rng)
                torch.save(W0, path / f"{i}.pt")

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @staticmethod
    def _generate(n_neurons: int, dist_params: DistributionParams, sparsity: float, rng) -> torch.Tensor:
        """Builds the internal structure of a cluster"""
        if dist_params.name == 'glorot':
            W0 = NormalConnectivityDataset._generate_glorot_w0(n_neurons, dist_params.mean, dist_params.std, rng)
        if dist_params.name == 'normal':
            W0 = NormalConnectivityDataset._generate_normal_w0(n_neurons, dist_params.mean, dist_params.std, rng)
        W0 = NormalConnectivityDataset._dales_law(W0)
        W0[torch.eye(n_neurons, dtype=torch.bool)] = 0
        W0[W0 != 0] = W0[W0 != 0] * (torch.rand_like(W0[W0 != 0]) > sparsity)
        return W0

    @staticmethod
    def _dales_law(W0: torch.Tensor) -> torch.Tensor:
        """Applies Dale's law to the connectivity matrix W0"""
        W0 = torch.concat((W0 * (W0 > 0), W0 * (W0 < 0)), 0)
        return W0

    @staticmethod
    def _generate_normal_w0(n_neurons: int, mean: float, std: float, rng) -> torch.Tensor:
        """Generates a normal n_neurons/2*n_neurons/2 matrux from a normal distribution"""
        half_n_neurons = n_neurons // 2
        W0 = torch.normal(mean, std, (half_n_neurons, n_neurons), generator=rng)
        return W0

    @staticmethod 
    def _generate_glorot_w0(n_neurons: int, mean: float, std: float, rng) -> torch.Tensor:
        """Generates a normal n_neurons/2*n_neurons/2 matrux from a normal distribution"""
        normal_W0 = NormalConnectivityDataset._generate_normal_w0(n_neurons, mean, std, rng)
        glorot_W0 = normal_W0 / torch.sqrt(torch.tensor(n_neurons, dtype=torch.float32))
        return glorot_W0

    @staticmethod
    def generate_examples(n_neurons: int, n_examples: int, dist_params: DistributionParams, seed=None, sparsity=0):
        """Generates a set of connectivity matrices W0"""
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        w0_list = []
        for i in range(n_examples):
            w0_square = NormalConnectivityDataset._generate(n_neurons, dist_params, sparsity, rng)
            num_neurons = w0_square.shape[0]
            non_zero_edges = w0_square.nonzero().t()
            edge_index, _ = add_remaining_self_loops(non_zero_edges, num_nodes=num_neurons)
            w0 = w0_square[edge_index[0], edge_index[1]]
            data = Data(edge_index=edge_index, num_nodes=num_neurons, W0=w0)
            w0_list.append(data)

        return w0_list


