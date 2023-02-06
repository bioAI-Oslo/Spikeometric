import torch
from spiking_network.datasets.connectivity_generator import ConnectivityGenerator

class UniformGenerator(ConnectivityGenerator):
    """
    A dataset of connectivity matrices for networks of neurons, where each matrix is generated from a
    uniform distribution over the range [low, high] and then sparsified to the specified sparsity.
    """
    def __init__(self, n_neurons: int, low: float, high: float, sparsity: float, rng=None) -> None:
        """Generates a dataset of uniformly distributed connectivity matrices, and saves them to the raw directory if 
        they do not already exist. If the raw directory already exists, the dataset is loaded from the raw directory and
        """
        self.n_neurons = n_neurons
        self.low = low
        self.high = high
        self.sparsity = sparsity
        self.rng = rng if rng is not None else torch.Generator()

    def generate_W0(self):
        """Generates a single connectivity matrix W0 from a uniform distribution over the range [low, high] and then sparsifies it to the specified sparsity"""""
        w0 = torch.rand(size=(self.n_neurons, self.n_neurons), generator=self.rng)*(self.high - self.low) + self.low # Uniform distribution over [low, high]
        w0[torch.rand((self.n_neurons, self.n_neurons)) < self.sparsity] = 0 
        w0[torch.eye(self.n_neurons, dtype=torch.bool)] = 0 # Set diagonal to zero
        return w0
    