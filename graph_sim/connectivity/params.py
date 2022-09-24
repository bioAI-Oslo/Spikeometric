from dataclasses import dataclass

@dataclass
class NormalParams:
    mean: float
    std: float

@dataclass
class FilterParams:
    ref_scale: int
    abs_ref_scale: int
    spike_scale: int
    abs_ref_strength: float
    rel_ref_strength: float
    decay_offdiag: float
    decay_diag: float

