from dataclasses import dataclass
from typing import Dict, Any
from dataclasses import asdict

@dataclass
class XRRConfig:
    tt_min: float = 0.10
    tt_max: float = 1.0
    gpos1: float = -0.043
    gpos2: float = 0.027
    lpos1: float = -25
    lpos2: float = 25
    num_points: int = 64

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
