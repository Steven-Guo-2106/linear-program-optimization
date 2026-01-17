from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class LPStatus(Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"

@dataclass
class LPProblem:
    obj: str                # "max" or "min"
    c: np.ndarray           # (n,)
    A: np.ndarray           # (m, n)
    b: np.ndarray           # (m,)
    constraint_signs: list  # length m, consists of {"<=", ">=", "="}
    var_bounds: list        # length n, consists of {">=0", "<=0", "free"}
    obj_const: float = 0.0

@dataclass
class SimplexResult:
    status: LPStatus
    x: Optional[np.ndarray] = None
    objective_value: Optional[float] = None
    y: Optional[np.ndarray] = None
    ray: Optional[np.ndarray] = None
    basis: Optional[np.ndarray] = None

