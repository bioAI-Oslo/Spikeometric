from dataclasses import dataclass

@dataclass
class DistibutionParams:
    """Class for storing distribution parameters."""

@dataclass
class NormalParams(DistibutionParams):
    mean: float = 0.0
    std: float = 5.0
    name: str = "normal"

@dataclass
class GlorotParams(DistibutionParams):
    mean: float = 0.0
    std: float = 5.0
    name: str = "glorot"

@dataclass
class FilterParams:
    """Filter parameters for the network"""
    n_neurons: int
    dist_params: DistibutionParams = GlorotParams()
    n_clusters: int = 1
    n_hub_neurons: int = 0
    ref_scale = 10
    abs_ref_scale = 3
    spike_scale = 5
    abs_ref_strength = -100
    rel_ref_strength = -30
    decay_offdiag = 0.2
    decay_diag = 0.5
    threshold = -5
