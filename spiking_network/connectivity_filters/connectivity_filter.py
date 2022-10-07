from spiking_network.connectivity_filters.abstract_connectivity_filter import AbstractConnectivityFilter
import torch

class ConnectivityFilter(AbstractConnectivityFilter):
    def __init__(self, W0, T, filter_params):
        super().__init__(W0, T, filter_params)

    def time_dependence(self, W0, t, filter_params, i, j):
        r"""Determines the time-dependendence of the connection between neuorns i, j"""
        diag = (i==j).unsqueeze(1).repeat(1, self.time_scale)
        diag_values = filter_params["abs_ref_strength"] * (t < 3) + filter_params["rel_ref_strength"] * torch.exp(-filter_params["beta"] * (t - 3)) * (3 <= t)
        off_diag_values = torch.einsum("i, ij -> ij", W0, torch.exp(-filter_params["alpha"] * t) * (t < 5))

        return torch.where(diag, diag_values, off_diag_values).flip(1)
