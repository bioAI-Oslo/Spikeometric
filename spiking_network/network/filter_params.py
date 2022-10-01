from dataclasses import dataclass

@dataclass
class DistributionParams:
    """Class for storing distribution parameters."""

    def _to_dict(self):
        return self.__dict__

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

    def _to_dict(self):
        return {
            "dist_params": self.dist_params._to_dict(),
            "ref_scale": self.ref_scale,
            "abs_ref_scale": self.abs_ref_scale,
            "spike_scale": self.spike_scale,
            "abs_ref_strength": self.abs_ref_strength,
            "rel_ref_strength": self.rel_ref_strength,
            "decay_offdiag": self.decay_offdiag,
            "decay_diag": self.decay_diag,
            "threshold": self.threshold,
            }
