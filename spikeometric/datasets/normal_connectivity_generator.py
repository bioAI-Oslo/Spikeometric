import torch
from spikeometric.datasets.connectivity_generator import ConnectivityGenerator
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj
from typing import List

class NormalGenerator(ConnectivityGenerator):
    r"""
    Generates a dataset of connectivity matrices W0 with an equal number of excitatory and inhibitory neurons
    with a normal distribution of weights.

    Example:
        >>> from spikeometric.datasets import NormalGenerator, ConnectivityDataset
        >>> generator = NormalGenerator(n_neurons=6, mean=0, std=1, sparsity=0.5, glorot=True)
        >>> generator.save(10, "data/w0/6_neurons_10_networks_0_mean_1_std_0.5_sparsity_glorot_0_seed")
        >>> dataset = ConnectivityDataset("data/w0/6_neurons_10_networks_0_mean_1_std_0.5_sparsity_glorot_0_seed")
        >>> data = dataset[0]
        >>> data
        Data(edge_index=[2, 13], num_nodes=6, W0=[13], stimulus_mask=[6])
        >>> data.W0
        tensor([ 0.6270,  0.5048,  0.4300, -0.7440, -0.2918, -1.1955, -0.4468,  0.0000,
                 0.0000,  0.0000,  0.0000,  0.0000,  0.0000])
        >>> data.edge_index
        tensor([[0, 0, 1, 3, 3, 4, 5, 0, 1, 2, 3, 4, 5],
                [2, 4, 3, 1, 5, 5, 4, 0, 1, 2, 3, 4, 5]])
        >>> data.stimulus_mask
        tensor([False, False, False, False, False, False])

    Parameters:
    ------------
    n_neurons (int):
        The number of neurons in a network (must be even)
    mean (float):
        The mean of the normal distribution from which the weights are drawn
    std (float):
        The standard deviation of the normal distribution
    sparsity (float):
        For each edge, the probability that it will be set to 0
    glorot (bool):
        If True, the weights will be divided by the square root of the number of neurons
    rng (torch.Generator):
        The random number generator to use
    """
    def __init__(self, n_neurons: int, mean: float, std:float, sparsity=0.5, glorot=False, rng=None):
        if n_neurons % 2 != 0 or n_neurons < 2:
            raise ValueError("n_neurons must be positive and even to have as many excitatory as inhibitory neurons")
        if sparsity < 0.5 or sparsity > 1.0:
            raise ValueError("sparsity must be between 0.5 and 1.0 (The number of edges is already halfed by Dale's law)")

        self.n_neurons = n_neurons
        self.mean = mean
        self.std = std
        self.sparsity = sparsity
        self.glorot = glorot
        self.rng = rng if rng is not None else torch.Generator()

    def generate_W0(self):
        r"""
        Generates a connectivity matrix W0 of size `n_neurons x n_neurons` with a normal distribution of weights 
        where Dale's law is applied to generate an equal number of excitatory and inhibitory neurons.
        Sparsity is applied by setting a fraction of the edges to 0.
        """
        half_n_neurons = self.n_neurons // 2
        half_W0 = torch.normal(self.mean, self.std, (half_n_neurons, self.n_neurons), generator=self.rng)
        if self.glorot:
            half_W0 = half_W0 / torch.sqrt(torch.tensor(self.n_neurons))
        half_W0[(torch.rand_like(half_W0) > 2*(1 - self.sparsity))] = 0
        W0 = torch.concat((half_W0 * (half_W0 > 0), half_W0 * (half_W0 < 0)), 0) # Dale's law
        W0[torch.eye(self.n_neurons, dtype=torch.bool)] = 0
        return W0