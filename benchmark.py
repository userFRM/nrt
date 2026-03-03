import numpy as np
from nrt.session import NRTSession
from nrt.models import zero_prior, delta_prior, spectral_prior
from nrt import NRTConfig

rng = np.random.default_rng(42)

# Data: a 512x512 matrix with realistic structure (low-rank signal + noise)
def make_data(seed=42):
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, (512, 20))
    B = rng.normal(0, 1, (20, 512))
    signal = (A @ B) * 0.1
    noise = rng.normal(0, 0.01, signal.shape)
    return (signal + noise).astype(np.float32)

data = make_data()
config = NRTConfig(epsilon=0.01, components_per_round=0.15, max_rounds=30)
sess = NRTSession()

scenarios = [
    ("Zero prior (no knowledge)",    None),
    ("Delta prior (80% similar)",     delta_prior(data, noise_std=0.3)),
    ("Spectral prior (top 20% SVD)", spectral_prior(data, keep_fraction=0.2)),
]

print(f"{'Scenario':<35} {'Rounds':>7} {'Bytes':>10} {'% of raw':>9} {'Converged':>10}")
print("-" * 75)
for name, prior in scenarios:
    r = sess.run(data, prior=prior, loss_rate=0.0, config=config)
    pct = r.total_bytes / r.data_size_bytes * 100
    print(f"  {name:<33} {r.rounds:>7} {r.total_bytes:>10} {pct:>8.1f}% {str(r.converged):>10}")

print()
print("Loss resilience (zero prior, 30% packet loss):")
r_loss = sess.run(data, prior=None, loss_rate=0.30, config=config)
r_clean = sess.run(data, prior=None, loss_rate=0.00, config=config)
print(f"  Clean channel:  {r_clean.rounds} rounds, {r_clean.total_bytes} bytes")
print(f"  30% loss:       {r_loss.rounds} rounds, {r_loss.total_bytes} bytes")
print(f"  Both converged: {r_clean.converged and r_loss.converged}")
print()
print("Retransmit norms (loss scenario):")
if r_loss.retransmit_norms:
    for orig, retx in r_loss.retransmit_norms[:3]:
        ratio = retx / orig if orig > 0 else 0
        print(f"  ||original||={orig:.4f}  ||retransmit||={retx:.4f}  ratio={ratio:.3f}  ({'✓ smaller' if retx <= orig else '✗ larger'})")
