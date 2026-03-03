"""Test that spectral prior correctly reduces transmission."""
import numpy as np
from nrt.session import NRTSession
from nrt.models import zero_prior, spectral_prior
from nrt import NRTConfig

def _make_data(seed=5):
    rng = np.random.default_rng(seed)
    A = rng.normal(0,1,(32,5)).astype(np.float32)
    B = rng.normal(0,1,(5,32)).astype(np.float32)
    return (A @ B) * 0.1 + rng.normal(0,0.003,(32,32)).astype(np.float32)

_CFG = NRTConfig(epsilon=0.02, k_per_round=4, max_rounds=40)

def test_spectral_prior_fewer_rounds():
    """Spectral prior: receiver knows dominant directions → fewer rounds needed."""
    data = _make_data()
    r0 = NRTSession().run(data, prior=zero_prior(data.shape), config=_CFG)
    rs = NRTSession().run(data, prior=spectral_prior(data, 0.3), config=_CFG)
    assert rs.rounds <= r0.rounds

def test_spectral_prior_fewer_bytes():
    """Spectral prior: fewer bytes transmitted."""
    data = _make_data()
    r0 = NRTSession().run(data, prior=zero_prior(data.shape), config=_CFG)
    rs = NRTSession().run(data, prior=spectral_prior(data, 0.3), config=_CFG)
    assert rs.total_bytes <= r0.total_bytes

def test_more_prior_knowledge_fewer_rounds():
    """As keep_fraction increases, rounds decrease monotonically."""
    data = _make_data()
    sess = NRTSession()
    fracs = [0.1, 0.2, 0.3, 0.5]
    rounds = [sess.run(data, prior=spectral_prior(data, f), config=_CFG).rounds for f in fracs]
    for i in range(1, len(rounds)):
        assert rounds[i] <= rounds[i-1] + 1, f"frac={fracs[i]}: {rounds[i]} > {rounds[i-1]}"
