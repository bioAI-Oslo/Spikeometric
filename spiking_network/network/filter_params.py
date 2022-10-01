from dataclasses import dataclass

@dataclass
class DistributionParams:
    """Class for storing distribution parameters."""

@dataclass
class NormalParams(DistributionParams):
    mean: float = 0.0
    std: float = 5.0
    name: str = "normal"

@dataclass
class GlorotParams(DistributionParams):
    mean: float = 0.0
    std: float = 5.0
    name: str = "glorot"

@dataclass
class FilterParams:
    """Filter parameters for the network"""
    dist_params: DistributionParams = GlorotParams()
    ref_scale = 10
    abs_ref_scale = 3
    spike_scale = 5
    abs_ref_strength = -100
    rel_ref_strength = -30
    decay_offdiag = 0.2
    decay_diag = 0.5
    threshold = -5

    def __eq__(self, other) -> bool:
        return self.__dict__ == other.__dict__
