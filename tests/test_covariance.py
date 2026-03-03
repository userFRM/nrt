"""Test that belief covariance correctly represents and updates uncertainty."""
import numpy as np
from nrt.belief import BeliefState
from nrt.models import zero_prior, spectral_prior

def test_zero_prior_uniform_uncertainty():
    """Zero prior: all directions have equal uncertainty (Σ_L = I, Σ_R = I)."""
    b = zero_prior((8, 8))
    assert np.allclose(b.left_cov, np.eye(8), atol=1e-5)
    assert np.allclose(b.right_cov, np.eye(8), atol=1e-5)

def test_spectral_prior_known_directions_zero_uncertainty():
    """
    Spectral prior: known SVD directions have λ = 0 in the covariance.
    The eigenbasis should not include those directions.
    """
    rng = np.random.default_rng(1)
    A = rng.normal(0,1,(16,3)).astype(np.float32)
    B = rng.normal(0,1,(3,16)).astype(np.float32)
    data = A @ B
    b = spectral_prior(data, keep_fraction=0.3)
    # known left directions: U_p. Their uncertainty should be near 0.
    U, s, Vt = np.linalg.svd(data, full_matrices=False)
    p = max(1, int(min(16,16)*0.3))
    for i in range(p):
        u = U[:, i]
        lam = float(u @ b.left_cov @ u)
        assert lam < 1e-4, f"Known left direction {i} has uncertainty {lam}, expected ~0"

def test_covariance_decreases_after_correction():
    """After applying a correction, uncertainty in that direction must decrease."""
    b = zero_prior((8, 8))
    from nrt import CorrectionFrame
    u = np.eye(8)[0].astype(np.float32)  # first basis vector
    v = np.eye(8)[0].astype(np.float32)
    frame = CorrectionFrame(u=u, v=v, c=1.0, lambda_ij=1.0, round_idx=0)
    lam_before = float(u @ b.left_cov @ u)
    b.update_covariance([frame])
    lam_after = float(u @ b.left_cov @ u)
    assert lam_after < lam_before

def test_uncertainty_eigenbasis_skips_known():
    """Known directions (λ≈0) must not appear in uncertainty eigenbasis."""
    rng = np.random.default_rng(2)
    data = rng.normal(0,1,(12,12)).astype(np.float32)
    b = spectral_prior(data, keep_fraction=0.4)
    directions = b.uncertainty_eigenbasis(k=20)
    U, _, Vt = np.linalg.svd(data, full_matrices=False)
    p = max(1, int(12*0.4))
    for u_dir, v_dir, lij in directions:
        # None of the returned directions should align with known left directions
        for i in range(p):
            overlap = abs(float(U[:, i] @ u_dir))
            assert overlap < 0.5, f"Known direction {i} appeared in eigenbasis (overlap={overlap:.3f})"
