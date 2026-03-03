"""Test retransmission properties."""
import numpy as np, copy
from nrt.session import NRTSession
from nrt.models import zero_prior
from nrt.sender import NRTSender
from nrt import NRTConfig

def _make_data(seed=3):
    rng = np.random.default_rng(seed)
    A = rng.normal(0,1,(32,5)).astype(np.float32)
    B = rng.normal(0,1,(5,32)).astype(np.float32)
    return (A @ B) * 0.1 + rng.normal(0,0.003,(32,32)).astype(np.float32)

_CFG = NRTConfig(epsilon=0.02, k_per_round=4, max_rounds=40)

def test_retransmit_norm_leq_original():
    """||retransmit|| ≤ ||original|| — guaranteed by monotone convergence."""
    data = _make_data()
    r = NRTSession().run(data, prior=zero_prior(data.shape), loss_rate=0.3, config=_CFG)
    for orig, retx in r.retransmit_norms:
        assert retx <= orig + 1e-5, f"retx={retx:.4f} > orig={orig:.4f}"

def test_retransmit_is_not_copy():
    """Retransmit is computed against current belief — not identical to original."""
    data = _make_data()
    belief = zero_prior(data.shape)
    sender = NRTSender(data, _CFG)
    # Send one packet, then retransmit — belief changed, so corrections differ
    import copy
    p1 = sender._make_packet(copy.deepcopy(belief), 0)
    # Apply p1 to belief (simulate one correction arriving)
    belief.apply_correction(p1.frames)
    belief.update_covariance(p1.frames)
    # Retransmit targeting updated belief
    p2 = sender.retransmit(copy.deepcopy(belief), 0)
    c1 = np.array([f.c for f in p1.frames])
    c2 = np.array([f.c for f in p2.frames])
    assert not np.allclose(c1, c2), "Retransmit should differ from original"

def test_converges_under_loss():
    """30% loss: still converges, just slower."""
    data = _make_data()
    r = NRTSession().run(data, prior=zero_prior(data.shape), loss_rate=0.30, config=_CFG)
    assert r.converged
