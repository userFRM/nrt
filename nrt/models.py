import numpy as np
from nrt.belief import BeliefState

def zero_prior(shape) -> BeliefState:
    """No knowledge. Σ_L = I, Σ_R = I. Degenerates to residual-SVD ordering."""
    return BeliefState(shape)

def spectral_prior(data: np.ndarray, keep_fraction: float = 0.20) -> BeliefState:
    """
    Receiver has the top-p SVD components of the data.
    Known directions have λ = 0 in the covariance.

    Σ_L = I - U_p U_p^T  (project OUT the known left directions)
    Σ_R = I - V_p V_p^T  (project OUT the known right directions)
    X̂   = U_p Σ_p V_p^T
    """
    m, n = data.shape
    p = max(1, int(min(m, n) * keep_fraction))
    U, s, Vt = np.linalg.svd(data.astype(np.float32), full_matrices=False)
    U_p, s_p, Vt_p = U[:, :p], s[:p], Vt[:p, :]

    estimate = (U_p * s_p) @ Vt_p
    # Left cov: identity minus projection onto known directions
    left_cov  = np.eye(m, dtype=np.float32) - U_p @ U_p.T
    right_cov = np.eye(n, dtype=np.float32) - Vt_p.T @ Vt_p

    return BeliefState(data.shape, estimate=estimate,
                       left_cov=left_cov, right_cov=right_cov)
