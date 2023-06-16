from typing import List
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops
from pathlib import Path
import torch

class ConnectivityGenerator(object):
    """Base class for generating connectivity matrices W0"""
    def generate(self, n_examples: int) -> List[Data]:
        """Generates a set of connectivity matrices W0 and returns them as a list of torch_geometric.data.Data objects"""
        w0_list = []
        for i in range(n_examples):
            W0_square = self.generate_W0()
            edge_index = W0_square.nonzero().t()
            W0 = W0_square[edge_index[0], edge_index[1]]
            data = Data(edge_index=edge_index, W0=W0, num_nodes=self.n_neurons)
            w0_list.append(data)
        return w0_list
    
    def save(self, n_networks, path):
        """Saves n_networks square connectivity matrices W0 to the specified path"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for i in range(n_networks):
            w0 = self.generate_W0()
            torch.save(w0, path / f"{i}.pt")