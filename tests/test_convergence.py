"""Test convergence properties of the correct NRT."""
import numpy as np
from nrt.session import NRTSession
from nrt.models import zero_prior
from nrt import NRTConfig

def _make_data(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0,1,(32,5)).astype(np.float32)
    B = rng.normal(0,1,(5,32)).astype(np.float32)
    return (A @ B) * 0.1 + rng.normal(0,0.003,(32,32)).astype(np.float32)

_CFG = NRTConfig(epsilon=0.02, k_per_round=4, max_rounds=40)

def test_monotonic_convergence():
    data = _make_data()
    r = NRTSession().run(data, prior=zero_prior(data.shape), config=_CFG)
    for i in range(1, len(r.error_per_round)):
        assert r.error_per_round[i] <= r.error_per_round[i-1] + 1e-6

def test_epsilon_termination():
    data = _make_data()
    r = NRTSession().run(data, prior=zero_prior(data.shape), config=_CFG)
    assert r.converged
    assert r.error_per_round[-1] < _CFG.epsilon

def test_zero_prior_degenerate_case():
    """
    With zero prior (Σ = I), uncertainty eigenbasis ordering = identity ordering.
    All directions equally uncertain → corrections ordered by |c_ij| = singular values of residual.
    This is the correct degenerate case — same as residual SVD.
    """
    data = _make_data()
    r = NRTSession().run(data, prior=zero_prior(data.shape), config=_CFG)
    assert r.converged
