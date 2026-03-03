from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class NRTConfig:
    epsilon: float = 0.01           # sufficiency threshold (relative Frobenius norm)
    k_per_round: int = 5            # directions transmitted per round
    max_rounds: int = 50

@dataclass
class CorrectionFrame:
    """One correction: a direction (u_i, v_j) and scalar c_ij."""
    u: np.ndarray        # left eigenvector (shape: m,)
    v: np.ndarray        # right eigenvector (shape: n,)
    c: float             # scalar correction: u^T (X - X̂) v
    lambda_ij: float     # uncertainty of this direction at time of send
    round_idx: int

@dataclass
class RefinementPacket:
    """One round's worth of corrections."""
    round_idx: int
    frames: List[CorrectionFrame]
    phi_after: float     # sender's estimate of residual after applying this
    is_final: bool

@dataclass
class SessionResult:
    error_per_round: List[float]
    bytes_per_round: List[int]
    total_bytes: int
    rounds: int
    converged: bool
    retransmit_norms: List[Tuple[float, float]]
    data_size_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.total_bytes / self.data_size_bytes if self.data_size_bytes else 0
