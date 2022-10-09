from spiking_network.connectivity_filters.abstract_connectivity_filter import AbstractConnectivityFilter
import torch

class ConnectivityFilter(AbstractConnectivityFilter):
    def time_dependence(self, W0, i, j, filter_params):
        r"""Determines the time-dependendence of the connection between neuorns i, j"""
        time_scale = 10
        abs_ref_scale = 3
        rel_ref_scale = 7
        influence_scale = 5

        t = torch.arange(time_scale).repeat(W0.shape[0], 1)
        is_self_edge = (i==j).unsqueeze(1).repeat(1, time_scale)
        self_edges = filter_params["abs_ref_strength"] * (t < abs_ref_scale) + filter_params["rel_ref_strength"] * torch.exp(-filter_params["beta"] * (t - abs_ref_scale)) * (abs_ref_scale <= t) * (t <= abs_ref_scale + rel_ref_scale)
        other_edges = torch.einsum("i, ij -> ij", W0, torch.exp(-filter_params["alpha"] * t) * (t < influence_scale))

        return torch.where(is_self_edge, self_edges, other_edges).flip(1) # Flip to get latest time step last
