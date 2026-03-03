import numpy as np
from typing import List, Tuple

class BeliefState:
    """
    Receiver's full belief state.

    Maintains estimate X̂ and Kronecker-structured covariance Σ_L ⊗ Σ_R.
    Covariance represents uncertainty: high λ_ij = don't know this direction.
    """

    def __init__(self, shape: Tuple[int, int], estimate: np.ndarray = None,
                 left_cov: np.ndarray = None, right_cov: np.ndarray = None):
        m, n = shape
        self.shape = shape
        self.estimate = estimate if estimate is not None else np.zeros(shape, dtype=np.float32)
        # Start with identity covariance = uncertain about all directions equally
        self.left_cov  = left_cov  if left_cov  is not None else np.eye(m, dtype=np.float32)
        self.right_cov = right_cov if right_cov is not None else np.eye(n, dtype=np.float32)

    def uncertainty_eigenbasis(self, k: int):
        """
        Return the top-k directions (u_i, v_j, λ_ij) sorted by λ_ij descending.

        Uses Kronecker structure: λ_ij = λ_i^L * λ_j^R
        Directions with λ_ij = 0 are KNOWN — skip them.
        """
        # Eigendecompose each factor
        lam_L, V_L = np.linalg.eigh(self.left_cov)   # ascending order
        lam_R, V_R = np.linalg.eigh(self.right_cov)
        # Sort descending
        idx_L = np.argsort(lam_L)[::-1]
        idx_R = np.argsort(lam_R)[::-1]
        lam_L, V_L = lam_L[idx_L], V_L[:, idx_L]
        lam_R, V_R = lam_R[idx_R], V_R[:, idx_R]

        # Build outer-product directions sorted by combined uncertainty
        directions = []
        for i, (li, ui) in enumerate(zip(lam_L, V_L.T)):
            for j, (lj, vj) in enumerate(zip(lam_R, V_R.T)):
                lij = float(li * lj)
                if lij > 1e-10:  # skip known directions
                    directions.append((ui, vj, lij))

        directions.sort(key=lambda x: x[2], reverse=True)
        return directions[:k]

    def cov_sqrt(self):
        """Return (Σ_L^{1/2}, Σ_R^{1/2}) via eigendecomposition."""
        lam_L, V_L = np.linalg.eigh(self.left_cov)
        sqrt_lam_L = np.sqrt(np.maximum(lam_L, 0))
        sqrt_L = (V_L * sqrt_lam_L) @ V_L.T

        lam_R, V_R = np.linalg.eigh(self.right_cov)
        sqrt_lam_R = np.sqrt(np.maximum(lam_R, 0))
        sqrt_R = (V_R * sqrt_lam_R) @ V_R.T

        return sqrt_L.astype(np.float32), sqrt_R.astype(np.float32)

    def apply_correction(self, frames):
        """Apply a list of CorrectionFrame to update estimate."""
        for frame in frames:
            self.estimate = self.estimate + frame.c * np.outer(frame.u, frame.v)

    def update_covariance(self, frames):
        """
        After receiving corrections for directions (u_i, v_j),
        reduce uncertainty: project out those directions from left/right covariance.

        For each corrected left direction u_i: Σ_L -= λ_i * u_i u_i^T
        For each corrected right direction v_j: Σ_R -= λ_j * v_j v_j^T
        """
        for frame in frames:
            u = frame.u.reshape(-1, 1)
            v = frame.v.reshape(-1, 1)
            # Project out this direction from left covariance
            lam_u = (u.T @ self.left_cov @ u).item()
            if lam_u > 1e-10:
                self.left_cov -= lam_u * (u @ u.T)
                self.left_cov = np.clip(self.left_cov, 0, None)  # numerical stability
            # Project out this direction from right covariance
            lam_v = (v.T @ self.right_cov @ v).item()
            if lam_v > 1e-10:
                self.right_cov -= lam_v * (v @ v.T)
                self.right_cov = np.clip(self.right_cov, 0, None)

    def residual_norm(self, truth: np.ndarray) -> float:
        """Relative Frobenius error."""
        denom = np.linalg.norm(truth) + 1e-8
        return float(np.linalg.norm(truth - self.estimate) / denom)
