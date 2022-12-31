from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.utils  import add_remaining_self_loops, to_dense_adj
from pathlib import Path
import numpy as np
import os
import torch

class ConnectivityDataset(InMemoryDataset):
    def __init__(self, w0_list):
        super().__init__()
        if isinstance(w0_list[0], Data):
            self.data = w0_list
        else:
            for w0 in w0_list:
                if not isinstance(w0, torch.Tensor) and not isinstance(w0, np.ndarray):
                    raise ValueError("W0 must be a torch.Tensor or numpy.ndarray.")
                if w0.shape[0] != w0.shape[1]:
                    raise ValueError("W0 must be a square matrix.")
            self.data = self._from_adjacency_matrices(w0_list)
    
    @classmethod
    def load(cls, path):
        path = Path(path)
        w0_list = []
        for file in sorted(os.listdir(path)):
            data = np.load(path / file)
            w0_list.append(torch.from_numpy(data))
        return cls(w0_list)

    @classmethod
    def _from_adjacency_matrices(self, w0_list):
        if isinstance(w0_list[0], np.ndarray):
            w0_list = [torch.from_numpy(w0) for w0 in w0_list]
        
        data = []
        for w0 in w0_list:
            n_neurons = w0.shape[0]
            non_zero_edges = w0.nonzero().t()
            edge_index, _ = add_remaining_self_loops(non_zero_edges, num_nodes=n_neurons)
            w0 = w0[edge_index[0], edge_index[1]]
            data.append(Data(W0=w0, edge_index=edge_index, num_nodes=n_neurons))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def to_dense(self):
        dense = []
        for data in self.data:
            dense.append(
                to_dense_adj(data.edge_index, edge_attr=data.W0, max_num_nodes=data.num_nodes)[0]
            )
        return dense
