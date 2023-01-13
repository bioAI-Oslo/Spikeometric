from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj
from pathlib import Path
import numpy as np
import os
import torch
from typing import Union, List, Tuple


class ConnectivityDataset(InMemoryDataset):
    r"""
    A dataset of connectivity matrices for networks of neurons.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in a
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    Example:
        >>> from spiking_network.datasets import ConnectivityDataset
        >>> dataset = ConnectivityDataset("root/datasets/example_dataset")
        >>> data = dataset[0]
        >>> data
        Data(edge_index=[2, 100], num_nodes=10, W0=[100])
    """
    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        path = Path(self.root) / "raw"
        w0_list = []
        for file in sorted(os.listdir(path)):
            if file.endswith(".npy"):
                w0_square = torch.from_numpy(np.load(path / file))
            elif file.endswith(".pt"):
                w0_square = torch.load(path / file)
            num_neurons = w0_square.shape[0]
            non_zero_edges = w0_square.nonzero().t()
            edge_index, _ = add_remaining_self_loops(
                non_zero_edges,
                num_nodes=num_neurons
            )
            w0 = w0_square[edge_index[0], edge_index[1]]
            data = Data(edge_index=edge_index, num_nodes=num_neurons, W0=w0)
            w0_list.append(data)

        data, slices = self.collate(w0_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return "data.pt"

    def to_dense(self):
        dense = []
        for data in self:
            dense.append(
                to_dense_adj(
                    data.edge_index,
                    edge_attr=data.W0,
                    max_num_nodes=data.num_nodes
                )[0]
            )
        return dense
