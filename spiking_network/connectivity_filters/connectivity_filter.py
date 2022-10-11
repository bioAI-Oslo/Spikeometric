from spiking_network.connectivity_filters.abstract_connectivity_filter import AbstractConnectivityFilter
import torch

class ConnectivityFilter(AbstractConnectivityFilter):
    def __init__(self, W0):
        """Defines the parameters of the connectivity filter"""
        self.time_scale = 10
        self.abs_ref_scale = 3
        self.rel_ref_scale = 7
        self.influence_scale = 5
        self.alpha = 0.2
        self.beta = 0.5
        self.abs_ref_strength = -100.0
        self.rel_ref_strength = -30.0
        
        super().__init__(W0)

    def time_dependence(self, W0, i, j):
        r"""Determines the time-dependendence of the connection between neuorns i, j"""
        t = torch.arange(self.time_scale).repeat(W0.shape[0], 1) # Time steps
        is_self_edge = (i==j).unsqueeze(1).repeat(1, self.time_scale) # Is the edge a self-edge?
        self_edges = (
                self.abs_ref_strength * (t < self.abs_ref_scale) + 
                self.rel_ref_strength * torch.exp(-self.beta * (t - self.abs_ref_scale)) * (self.abs_ref_scale <= t) * (t <= self.abs_ref_scale + self.rel_ref_scale)
            ) # values for self-edges
        other_edges = torch.einsum("i, ij -> ij", W0, torch.exp(-self.alpha * t) * (t < self.influence_scale)) # values for other edges

        return torch.where(is_self_edge, self_edges, other_edges).flip(1) # Flip to get latest time step last

    @property
    def parameters(self):
        return {
                "time_scale": self.time_scale,
                "abs_ref_scale": self.abs_ref_scale,
                "rel_ref_scale": self.rel_ref_scale,
                "influence_scale": self.influence_scale,
                "alpha": self.alpha,
                "beta": self.beta,
                "abs_ref_strength": self.abs_ref_strength,
                "rel_ref_strength": self.rel_ref_strength,
            }

