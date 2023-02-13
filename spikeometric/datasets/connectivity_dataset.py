import os
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_remaining_self_loops

class ConnectivityDataset:
    r"""
    A dataset of connectivity matrices for networks of neurons.
    
    The connectivity matrices are loaded from a directory of .npy or .pt files.
    Each file should contain a square connectivity matrix for a network of neurons.
    The connectivity matrices are converted to torch_geometric `Data` objects 
    with `edge_index`, `W0`, `num_nodes` and `stimulus_mask` attributes.

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
        Data(edge_index=[2, 5042], W0=[5042], num_nodes=100, stimulus_mask=[100])
        >>> loader = DataLoader(dataset, batch_size=2)
        >>> for batch in loader:
        ...     print(batch)
        >>> for batch in loader:
        ...     print(batch)
        ...
        DataBatch(edge_index=[2, 25242], W0=[25242], num_nodes=500, stimulus_mask=[500], batch=[500], ptr=[6])
        DataBatch(edge_index=[2, 25250], W0=[25250], num_nodes=500, stimulus_mask=[500], batch=[500], ptr=[6])
    
    Parameters
    ----------
    root (string):
        Root directory where the dataset should be saved.
    add_self_loops (bool, optional):
        Whether to add self-loops to the connectivity matrices. (default: :obj:`True`)
    stimulus_masks (list, optional):
        A list of boolean masks for each connectivity matrix, indicating which neurons should be stimulated. (default: :obj:`[]`)
    """
    def __init__(self, root, stimulus_masks=[]):
        self.root = root
        self.stimulus_masks = stimulus_masks
        self.data = self.process()

    def process(self):
        """Processes the connectivity matrices in the root directory and returns a list of torch_geometric Data objects."""
        path = Path(self.root)
        files = os.listdir(path)

        if "stimulus_masks.pt" in files:
            self.stimulus_masks = torch.load(path / "stimulus_masks.pt")
            files.remove("stimulus_masks.pt")
        
        w0_list = []
        for i, file in enumerate(sorted(files)):
            # Check if file is a .npy or .pt file and load it
            if file.endswith(".npy"):
                w0_square = torch.from_numpy(np.load(path / file))
            elif file.endswith(".pt"):
                w0_square = torch.load(path / file)
            
            # Convert the connectivity matrix to a sparse adjacency matrix
            num_neurons = w0_square.shape[0]
            edge_index = w0_square.nonzero().t()
            w0 = w0_square[edge_index[0], edge_index[1]]

            # Create a boolean mask of the target neurons
            stimulus_mask = torch.zeros(num_neurons, dtype=torch.bool)
            if len(self.stimulus_masks) > 0:
                stimulus_mask[self.stimulus_masks[i].squeeze()] = True
            
            # Create a torch_geometric Data object and add it to the list
            data = Data(edge_index=edge_index, num_nodes=num_neurons, W0=w0, stimulus_mask=stimulus_mask)
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

    def add_stimulus_masks(self, stimulus_masks):
        """Adds a set of stimulus masks to the dataset."""
        self.stimulus_masks = stimulus_masks
        self.data = self.process()