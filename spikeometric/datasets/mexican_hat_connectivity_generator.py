import torch
from spikeometric.datasets.connectivity_generator import ConnectivityGenerator

class MexicanHatGenerator(ConnectivityGenerator):
    r"""
    A dataset of connectivity matrices for ring-networks of neurons, with weights generated from a Mexican hat distribution.
    """
    def __init__(self, n_neurons: float, a: float, sigma_1: float, sigma_2: float) -> None:
        """Constructs a MexicanHatGenerator object. If the raw directory does not exist, it will be created and populated"""
        self.n_neurons = n_neurons
        self.a = a
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

    def generate_W0(self):
        r"""
        Generate a ring network with a Mexican hat distribution of weights between neurons. The weights follow the
        equation:

        .. math::
            W_{ij} = \exp\left(-\frac{d_{i,j}^2}{2\sigma_1^2}\right) - a\exp\left(-\frac{d{i,j}^2}{2\sigma_2^2}\right)
        
        with :math:`d_{i,j}` the distance between neurons :math:`i` and :math:`j` on the ring in number of neurons.
        """
        i,  j = torch.meshgrid(torch.arange(self.n_neurons), torch.arange(self.n_neurons), indexing='ij')
        d_ij = torch.min(torch.abs(i - j), self.n_neurons - torch.abs(i - j))
        W0 = torch.exp(-0.5*(d_ij/self.sigma_1)**2) - self.a*torch.exp(-0.5*(d_ij/self.sigma_2)**2)
        W0[torch.eye(self.n_neurons, dtype=torch.bool)] = 0
        return W0