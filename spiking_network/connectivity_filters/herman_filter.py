from spiking_network.connectivity_filters.abstract_connectivity_filter import AbstractConnectivityFilter

class HermanFilter(AbstractConnectivityFilter):
    def __init__(self, W0):
        super().__init__(W0)

    def time_dependence(self, W0, i, j):
        r"""Determines the time-dependendence of the connection between neuorns i, j"""
        return W0.unsqueeze(1)

    @property
    def parameters(self):
        return {}

