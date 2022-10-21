from spiking_network.connectivity_filters.abstract_connectivity_filter import (
    AbstractConnectivityFilter,
)


class HermanFilter(AbstractConnectivityFilter):
    def __init__(
        self,
        W0,
        N=100,
        dt=0.0001,
        tau=0.01,
        sig_1=6.98,
        sig_2=7,
        a1=1,
        a2=1.0005,
        b=0.001,  # uniform feedforward input
        noise_sd=0.3,  # amplitude of feedforward noise
        noise_sparsity=(
            1.5  # noise is injected with the prob that a standard normal exceeds this
        ),
        nsteps=int(4e5),  # no. of time-steps
    ):
        super().__init__(W0)

        self._parameters = {
            "N": N,
            "dt": dt,
            "tau": tau,
            "sig_1": sig_1,
            "sig_2": sig_2,
            "a1": a1,
            "a2": a2,
            "b": b,
            "noise_sd": noise_sd,
            "noise_sparsity": noise_sparsity,
            "nsteps": nsteps,
        }

    def time_dependence(self, W0, i, j):
        r"""Determines the time-dependendence of the connection between neuorns i, j"""
        return W0.unsqueeze(1)

    @property
    def parameters(self):
        return self._parameters
