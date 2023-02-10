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
