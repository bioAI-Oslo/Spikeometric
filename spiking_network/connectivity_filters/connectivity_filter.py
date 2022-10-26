import torch
import torch.nn as nn
from spiking_network.connectivity_filters.base_connectivity_filter import BaseConnectivityFilter

class ConnectivityFilter(BaseConnectivityFilter):
    def __init__(self, tuneable_parameters=[]):
        super().__init__()

        # Parameters
        parameters = {
            "alpha": 0.2,
            "beta": 0.5,
            "abs_ref_strength": -100.,
            "rel_ref_strength": -30.,
            "abs_ref_scale": 3,
            "rel_ref_scale": 7,
            "time_scale": 10,
            "influence_scale": 5,
        }

        self.params = nn.ParameterDict(
            {
                key: nn.Parameter(torch.tensor(value), requires_grad=True if key in tuneable_parameters else False)
                for key, value in parameters.items()
                }
        )

    def __call__(self, W0, edge_index):
        i, j = edge_index
        t = torch.arange(self.params["time_scale"], device=W0.device)
        t = t.repeat(W0.shape[0], 1) # Time steps
        is_self_edge = (i==j).unsqueeze(1).repeat(1, self.params["time_scale"]) # Is the edge a self-edge?

        # Compute the connectivity matrix
        self_edges = self._self_edges(t)
        other_edges = self._other_edges(W0, t)

        W = torch.where(is_self_edge, self_edges, other_edges).flip(1)

        return W

    def _self_edges(self, t):
        abs_ref = self.params["abs_ref_strength"] * (t < self.params["abs_ref_scale"])
        rel_ref = (
                self.params["rel_ref_strength"] * torch.exp(-torch.abs(self.params["beta"]) * (t - self.params["abs_ref_scale"]))
                * (self.params["abs_ref_scale"] <= t) * (t <= self.params["abs_ref_scale"] + self.params["rel_ref_scale"])
            )
        return abs_ref + rel_ref

    def _other_edges(self, W0, t):
        return (
                torch.einsum("i, ij -> ij", W0, torch.exp(-torch.abs(self.params["alpha"]) * t)
                * (t < self.params["influence_scale"]))
            )

