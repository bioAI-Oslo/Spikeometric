from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch

class ConnectivityDataset(InMemoryDataset):
    def __init__(self, transform=None):
        super().__init__(transform=transform)

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

    def _generate_data(self, n_neurons, n_datasets, seeds, **kwargs):
        datasets = []
        for i in range(n_datasets):
            W0 = self._generate(n_neurons, seeds[i], **kwargs)
            edge_index = W0.nonzero().t()
            W0 = W0[edge_index[0], edge_index[1]]
            datasets.append(Data(W0=W0, edge_index=edge_index, num_nodes=n_neurons))
        return datasets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def _dales_law(self, W0: torch.Tensor) -> torch.Tensor:
        """Applies Dale's law to the connectivity matrix W0"""
        W0 = torch.concat((W0 * (W0 > 0), W0 * (W0 < 0)), 0)
        return W0
