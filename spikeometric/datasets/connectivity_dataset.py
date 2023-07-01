import os
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import glob

class ConnectivityDataset:
    r"""
    A dataset of connectivity matrices for networks of neurons.
    
    The connectivity matrices are loaded from a directory of .npy or .pt files.
    Each file should contain a square connectivity matrix for a network of neurons.
    The connectivity matrices are converted to torch_geometric `Data` objects 
    with `edge_index`, `W0` and `num_nodes` attributes.

    By using torch_geometric's `DataLoader`, the connectivity matrices can be batched together into a single
    graph, with each of the n_networks examples as an isolated subgraph.

    Example:
        >>> from spikeometric.datasets import ConnectivityDataset
        >>> from torch_geometric.loader import DataLoader
        >>> dataset = ConnectivityDataset("datasets/example_dataset")
        >>> len(dataset)
        10
        >>> data = dataset[0]
        >>> data
        Data(edge_index=[2, 5042], W0=[5042], num_nodes=100)
        >>> loader = DataLoader(dataset, batch_size=2)
        >>> for batch in loader:
        ...     print(batch)
        >>> for batch in loader:
        ...     print(batch)
        ...
        DataBatch(edge_index=[2, 25242], W0=[25242], num_nodes=500, batch=[500], ptr=[6])
        DataBatch(edge_index=[2, 25250], W0=[25250], num_nodes=500, batch=[500], ptr=[6])
    
    Parameters
    ----------
    root (string):
        Root directory where the dataset should be saved.
    """
    def __init__(self, root):
        self.root = root
        self.data = self.process()

    def process(self):
        """Processes the connectivity matrices in the root directory and returns a list of torch_geometric Data objects."""
        path = Path(self.root)
        files = list(path.glob("*.npy")) + list(path.glob("*.pt"))

        w0_list = []
        for i, file in enumerate(sorted(files)):
            # Check if file is a .npy or .pt file and load it
            if file.name.endswith(".npy"):
                w0_square = torch.from_numpy(np.load(file))
            elif file.name.endswith(".pt"):
                w0_square = torch.load(file)
            
            # Convert the connectivity matrix to a sparse adjacency matrix
            num_neurons = w0_square.shape[0]
            edge_index = w0_square.nonzero().t()
            w0 = w0_square[edge_index[0], edge_index[1]]

            # Create a torch_geometric Data object and add it to the list
            data = Data(edge_index=edge_index, num_nodes=num_neurons, W0=w0)
            w0_list.append(data)
        
        return w0_list

    def __getitem__(self, idx):
        """Returns the Data object at index idx."""
        return self.data[idx]

    def __len__(self):
        """Returns the number of Data objects in the dataset."""
        return len(self.data)

    def combine_all(self):
        """Combines all the Data objects into a single Data object."""
        data_loader = DataLoader(self.data, batch_size=len(self.data))
        return next(iter(data_loader))