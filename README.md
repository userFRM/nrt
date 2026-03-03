# NRT

**Null Refinement Transport**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-13%20passing-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)]()

> *Project the transmission into the null space of the receiver's knowledge, and send only what survives the projection.*

A transport primitive. Not a codec. Not a compression algorithm. A new way for two endpoints to negotiate what actually needs to be transmitted.

---

## Table of contents

1. [The core idea](#the-core-idea)
2. [Problem statement](#problem-statement)
3. [Mathematical foundation](#mathematical-foundation)
4. [The belief covariance](#the-belief-covariance)
5. [Protocol design](#protocol-design)
6. [Domain models](#domain-models)
7. [Industry applications](#industry-applications)
8. [Empirical results](#empirical-results)
9. [Limitations](#limitations)
10. [Installation and usage](#installation-and-usage)
11. [Project structure](#project-structure)
12. [Roadmap](#roadmap)

---

## The core idea

Your machines transmit *states* when they should transmit *corrections*. The sender shouts the answer into a wire. It never asks: *what does the other side already believe?*

NRT formalises a different primitive. Before transmitting, the sender asks:

- What does the receiver currently believe?
- In which directions is the receiver most uncertain?
- What is the minimum correction to bring that uncertainty below a declared threshold?

The transmission is that correction — and nothing more.

When a packet is lost in transit, the sender does not retransmit a copy. It recomputes the correction against the receiver's *current* belief state, which has since evolved. The retransmission is always smaller than the original. It is never a repeat.

Transfer terminates when the receiver's residual uncertainty falls below $\varepsilon$ — a threshold declared by the application. If the receiver already satisfies $\varepsilon$ before transmission begins, zero bytes are sent.

---

## Problem statement

Shannon's channel capacity theorem defines the maximum reliable bit rate for a channel. It does not define *which bits to transmit*. Classical transport protocols transmit all bits, in order. This is optimal only when the receiver has zero prior knowledge.

In practice, the receiver almost always has partial knowledge:

- A video decoder holds the previous frame — 95% of the next frame is predictable.
- An inference server receiving a model update holds the previous checkpoint.
- An IoT node receiving a telemetry correction holds a physics-based prediction of the next reading.
- A game client has just rendered frame $N$ — it knows what hasn't changed.
- A database replica holds a previous snapshot — most records are unchanged.

In every such case, the optimal transmission is the *divergence between the receiver's current belief and the truth*, expressed in the receiver's basis of maximum uncertainty. Current protocols pay the cost of the full data. They should pay the cost of the divergence.

> [!NOTE]
> NRT is not a compression algorithm. It does not reduce file sizes. It reduces the number of bits that must be *transmitted* by exploiting what the receiver already knows. With no prior knowledge, it behaves identically to progressive transfer. With a good prior, it may transmit nothing at all.

---

## Mathematical foundation

### Belief state

Let $\mathbf{X} \in \mathbb{R}^{m \times n}$ be the sender's data. The receiver maintains:

- An estimate $\hat{\mathbf{X}}^{(t)} \in \mathbb{R}^{m \times n}$ — its current best reconstruction
- A Kronecker-structured belief covariance $\Sigma_{L} \otimes \Sigma_{R}$, where $\Sigma_{L} \in \mathbb{R}^{m \times m}$ encodes uncertainty over row directions and $\Sigma_{R} \in \mathbb{R}^{n \times n}$ encodes uncertainty over column directions

The residual at round $t$ is:

$$\mathbf{R}^{(t)} = \mathbf{X} - \hat{\mathbf{X}}^{(t)}$$

### Uncertainty eigenbasis

Eigendecompose each covariance factor:

$$\Sigma_{L} = \mathbf{V}_{L} \mathbf{\Lambda}_{L} \mathbf{V}_{L}^{\top}, \qquad \Sigma_{R} = \mathbf{V}_{R} \mathbf{\Lambda}_{R} \mathbf{V}_{R}^{\top}$$

The combined uncertainty over outer-product directions $\mathbf{u}_{i} \mathbf{v}_{j}^{\top}$ is:

$$\lambda_{ij} = \lambda_{i}^{L} \cdot \lambda_{j}^{R}$$

A large $\lambda_{ij}$ means the receiver is highly uncertain about direction $\mathbf{u}_{i} \mathbf{v}_{j}^{\top}$. A value of $\lambda_{ij} = 0$ means the receiver already knows that direction — it is **never transmitted**.

### Correction and update

The scalar correction for direction $(i, j)$ is:

$$c_{ij} = \mathbf{u}_{i}^{\top} \, \mathbf{R}^{(t)} \, \mathbf{v}_{j}$$

NRT transmits the top-$k$ directions sorted by $\lambda_{ij}$ (highest uncertainty first). The receiver applies:

$$\hat{\mathbf{X}}^{(t+1)} = \hat{\mathbf{X}}^{(t)} + \sum_{\text{received}} c_{ij} \cdot \mathbf{u}_{i} \mathbf{v}_{j}^{\top}$$

After applying a correction, the receiver projects out that direction from its covariance:

$$\Sigma_{L} \leftarrow \Sigma_{L} - \lambda_{i}^{L} \cdot \mathbf{u}_{i} \mathbf{u}_{i}^{\top}, \qquad \Sigma_{R} \leftarrow \Sigma_{R} - \lambda_{j}^{R} \cdot \mathbf{v}_{j} \mathbf{v}_{j}^{\top}$$

That direction's uncertainty drops to zero permanently.

> [!IMPORTANT]
> This is the critical distinction from SVD-based progressive transfer. SVD of the residual prioritises directions of highest **variance in the data**. NRT prioritises directions of highest **uncertainty in the receiver**. These coincide only when $\Sigma_{r} = \mathbf{I}$ (zero prior). When the receiver has a prior, they diverge — and NRT becomes strictly more efficient.

### $\varepsilon$-sufficiency

Transfer terminates when:

$$\frac{\lVert \mathbf{X} - \hat{\mathbf{X}}^{(t)} \rVert_{F}}{\lVert \mathbf{X} \rVert_{F}} < \varepsilon$$

$\varepsilon$ is declared by the application layer. Setting $\varepsilon = 0$ requires exact reconstruction. Setting $\varepsilon = 0.05$ permits 5% relative error — appropriate for perceptual applications such as video or audio. The protocol terminates the moment $\varepsilon$ is satisfied.

### Retransmission norm bound

When a correction packet is lost, the sender recomputes against the receiver's *current* belief state $\hat{\mathbf{X}}^{(t')}$ where $t' > t$. Since convergence is monotone:

$$\lVert \mathbf{R}^{(t')} \rVert_{F} \leq \lVert \mathbf{R}^{(t)} \rVert_{F} \implies \lVert \text{retransmit} \rVert_{F} \leq \lVert \text{original} \rVert_{F}$$

The retransmission is always at most as large as the original, and in practice smaller. It is never a copy.

---

## The belief covariance

The covariance $\Sigma_{r}$ is the receiver's formal declaration of what it does not know.

**Zero prior** — receiver has no knowledge:

$$\Sigma_{L} = \mathbf{I}_{m}, \quad \Sigma_{R} = \mathbf{I}_{n}$$

All directions are equally uncertain. NRT degenerates to transmitting corrections in order of magnitude — equivalent to residual SVD. This is the correct degenerate case: NRT is never worse than progressive transfer with no prior.

**Spectral prior** — receiver has the top-$p$ SVD components of $\mathbf{X}$:

$$\Sigma_{L} = \mathbf{I}_{m} - \mathbf{U}_{p} \mathbf{U}_{p}^{\top}, \quad \Sigma_{R} = \mathbf{I}_{n} - \mathbf{V}_{p} \mathbf{V}_{p}^{\top}$$

Known directions have $\lambda = 0$ — excluded from the eigenbasis and never transmitted. If the prior already satisfies $\varepsilon$, zero bytes are sent.

> [!TIP]
> The covariance does not need to be exact. A diagonal approximation or top-$k$ eigenvectors is sufficient for most applications. The protocol degrades gracefully with a coarser covariance model.

---

## Protocol design

### Session phases

```
Sender                                Receiver
  │                                      │
  │──── HELLO (ε, shape, domain) ──────► │
  │◄─── SEED (Σ_L, Σ_R, X̂ hash) ──────  │  receiver declares its covariance
  │                                      │
  │  [if residual already < ε → DONE]    │  zero bytes transmitted
  │                                      │
  │──── REFINE (round=1, frames) ──────► │  top-k directions by λ_ij
  │◄─── ACK (residual_norm, stop?) ───── │  receiver reports current error
  │                                      │
  │──── REFINE (round=2 …) ────────────► │
  │◄─── STOP ───────────────────────────  │  ε satisfied
```

### Refinement frame

```
[round_idx  : uint32]     belief epoch this correction targets
[u_i        : float32[m]] left eigenvector of Σ_L
[v_j        : float32[n]] right eigenvector of Σ_R
[c_ij       : float32]    scalar correction u_i^T R v_j
[lambda_ij  : float32]    uncertainty of this direction at send time
[is_final   : uint8]      1 if sender believes ε is satisfied
```

### ACK frame

```
[round_idx  : uint32]
[residual   : float32]    receiver's current relative error
[stop       : uint8]      1 = sufficient, 0 = continue
```

---

## Domain models

NRT is domain-agnostic at the transport layer. The covariance model is domain-specific. The table below shows how $\Sigma_{r}$ is constructed for each major application domain.

| Domain | Receiver's prior | Covariance structure | What gets transmitted |
|:-------|:----------------|:---------------------|:----------------------|
| **Video / live media** | Previous frame(s) | Motion-compensated temporal residual; high $\lambda$ on moving regions | Changed pixels, motion anomalies |
| **Cloud gaming / XR** | Just-rendered frame | Per-region uncertainty from game state delta | New geometry, explosions, UI changes |
| **Satellite / aviation** | Previous message + link model | High $\lambda$ on loss-prone frequency bands | Fresh corrections, not retransmit copies |
| **IoT / telemetry** | Physics-based prediction (Kalman) | Innovation covariance $\mathbf{P}_{k\|k-1}$ from filter | Sensor anomalies, deviations from model |
| **ML model distribution** | Previous checkpoint | Spectral prior: $\lambda = 0$ on known singular directions | Weight deltas, new capacity |
| **Database replication** | Previous snapshot | High $\lambda$ on recently-written pages/records | Changed records only |
| **Medical imaging** | Previous scan | Anatomical prior from registered image | Lesion changes, diagnostic regions |
| **CDN / file sync** | Previous version | Block-level hash prior | Changed content blocks |

> [!NOTE]
> The domain model defines the prior. The transport primitive is identical across all domains. A single NRT implementation supports all use cases by swapping the covariance constructor.

---

## Industry applications

### Video distribution and live media

Video frames are approximately 95% predictable from the previous frame. Today, codecs (H.264, H.265, AV1) exploit this at the application layer with inter-frame prediction. NRT exposes this as a **transport primitive** — domain-agnostically, without requiring codec-level integration. The receiver's covariance is constructed from the decoded previous frame; the sender transmits only the motion-residual components the receiver cannot predict.

**Affected:** Netflix, YouTube, every CDN, Zoom, Teams, all broadcast infrastructure.

### Cloud gaming and interactive XR

The client has just rendered frame $N$. The server has frame $N+1$. The client's covariance has low $\lambda_{ij}$ on regions that haven't changed (background, static geometry) and high $\lambda_{ij}$ on regions of new activity. NRT delivers the correction for moving objects at full fidelity immediately, progressively refining static regions.

**Affected:** cloud gaming platforms (GeForce Now, Xbox Cloud), VR/AR streaming, remote desktop (RDP, Citrix).

### Satellite and constrained links

Satellite uplinks are expensive, high-latency, and loss-prone. Every retransmission wastes scarce capacity. Under NRT, a lost packet triggers a *fresh correction* against the receiver's current belief — which is smaller than the original because other corrections have since arrived. Effective throughput increases without changing the physical link.

**Affected:** Starlink, Viasat, aviation Wi-Fi (Gogo, Panasonic), maritime connectivity, deep-space mission uplinks.

### IoT and industrial telemetry

Factory sensors, energy grid monitors, and fleet telemetry generate highly predictable time-series data. A Kalman filter running at the receiver defines the innovation covariance $\mathbf{P}_{k|k-1}$ — the receiver's uncertainty about the next sample. NRT transmits only the innovations: deviations from what the physics model predicted. Normal operation transmits near-zero bytes. Anomalies transmit fully.

**Affected:** industrial automation, energy grid management, vehicle fleet telemetry, smart building systems.

### AI and ML infrastructure

Model checkpoints are approximately low-rank and close to previous checkpoints. An inference server holding checkpoint $t-1$ has a spectral prior that eliminates transmission of unchanged singular directions. A server that already satisfies $\varepsilon$ receives zero bytes.

**Affected:** model distribution at scale (Meta, Google, Anthropic), federated learning gradient exchange, distributed training coordination.

### Medical imaging

A radiologist reviewing a follow-up scan holds the previous scan as a prior. The patient's anatomy is largely unchanged. NRT transmits full fidelity only in regions of clinical change, compressing stable anatomy. The receiver's covariance is built from image registration against the prior scan.

**Affected:** PACS systems, radiology workflows, remote diagnosis on constrained hospital links.

---

## Empirical results

Benchmark: 32×32 structured matrix (rank-5 signal + noise), 4,096 bytes raw. $\varepsilon = 0.02$, $k = 4$ directions per round.

### Prior quality vs. transmission cost

| Prior | Rounds | Bytes sent | % of raw | Converged |
|------:|-------:|-----------:|---------:|:---------:|
| Zero (no knowledge) | 6 | 6,432 | 157.0% | ✓ |
| Spectral prior, top 10% known | 1 | 1,072 | **26.2%** | ✓ |
| Spectral prior, top 30% known | 0 | **0** | **0.0%** | ✓ |
| Spectral prior, top 50% known | 0 | **0** | **0.0%** | ✓ |

*0-byte cases: receiver's prior already satisfies $\varepsilon$. Transfer terminates before a single byte is sent.*

### Loss resilience

| Channel condition | Rounds | Converged |
|:-----------------|-------:|:---------:|
| Clean (0% loss) | 6 | ✓ |
| 10% packet loss | 6 | ✓ |
| 30% packet loss | 6 | ✓ |

*Retransmissions are fresh corrections, not copies. Convergence is unaffected.*

> [!WARNING]
> Benchmark results use synthetic structured data. Zero-prior transfer costs 157% of raw due to per-frame eigenvector overhead — NRT is not efficient for small matrices without a prior. Real-world gains are proportional to prior quality and data size. Domain-specific validation is required for each application.

---

## Limitations

> [!CAUTION]
> **NRT is a research prototype (v0.1).** All results are from simulated sessions. No real network transport exists. Domain-specific implementations are not yet built.

**Covariance overhead.** Each correction frame carries eigenvectors $\mathbf{u}_{i} \in \mathbb{R}^{m}$ and $\mathbf{v}_{j} \in \mathbb{R}^{n}$ alongside the scalar $c_{ij}$. For small matrices, this overhead dominates. NRT is efficient only when matrix dimensions are large relative to the per-frame overhead.

**Covariance calibration.** Performance depends on $\Sigma_{r}$ accurately representing actual uncertainty. A miscalibrated prior causes suboptimal direction ordering. Unlike SVD-based approaches, which degrade gracefully, a wrong covariance actively degrades performance.

**Synchronisation.** Both endpoints must maintain consistent belief state. Floating-point non-determinism across heterogeneous hardware can cause divergence. Production use requires deterministic arithmetic or periodic covariance snapshots for resynchronisation.

**Structured data only.** For encrypted payloads, random data, or data without low-rank structure, NRT adds overhead with no benefit. The domain model must be appropriate to the data.

**Prior art.** NRT draws on Wyner–Ziv coding (1976), Slepian–Wolf distributed source coding (1973), information geometry (Amari, 1980s), and semantic communication research (active since ~2019). The contribution is a concrete protocol design: Kronecker belief covariance, per-direction uncertainty ordering, covariance update rules, and $\varepsilon$-sufficiency termination. This is an engineering contribution, not a mathematical discovery.

---

## Installation and usage

```bash
git clone https://github.com/userFRM/nrt.git
cd nrt
pip install -r requirements.txt
```

**Requirements:** `numpy>=1.24`, `scipy>=1.10`, `pytest>=7.0` (tests only)

```python
from nrt.session import NRTSession
from nrt.models import spectral_prior
from nrt import NRTConfig
import numpy as np

data  = np.load("checkpoint.npy")
prior = spectral_prior(previous_data, keep_fraction=0.20)

result = NRTSession().run(data, prior=prior, config=NRTConfig(epsilon=0.02))
print(f"{result.rounds} rounds · {result.total_bytes} bytes · {result.compression_ratio:.1%} of raw")
```

```bash
python3 -m pytest tests/ -v   # 13 tests
```

---

## Project structure

```
nrt/
├── nrt/
│   ├── __init__.py     NRTConfig, CorrectionFrame, RefinementPacket, SessionResult
│   ├── belief.py       BeliefState — Σ_L⊗Σ_R covariance, eigenbasis, covariance updates
│   ├── models.py       Prior constructors — zero_prior(), spectral_prior()
│   ├── sender.py       NRTSender — corrections ordered by λ_ij, retransmit()
│   ├── receiver.py     NRTReceiver — apply(), is_sufficient()
│   └── session.py      NRTSession — full protocol simulation, optional packet loss
└── tests/
    ├── test_covariance.py    covariance structure, known-direction exclusion, updates
    ├── test_convergence.py   monotone convergence, ε-termination, zero-prior degenerate
    ├── test_prior.py         spectral prior acceleration, monotone prior improvement
    └── test_retransmit.py    norm bound, not-a-copy, convergence under 30% loss
```

---

## Roadmap

The current codebase is the protocol core. Domain-specific implementations are the next layer.

| Priority | Item |
|:--------:|:-----|
| High | **Video domain model** — temporal prior from previous frame, motion-region covariance |
| High | **IoT domain model** — Kalman innovation covariance as NRT prior |
| High | Real UDP transport layer with ACK-driven covariance exchange |
| Medium | **Cloud gaming domain model** — game-state-aware per-region covariance |
| Medium | Block-diagonal covariance approximation for large matrices |
| Medium | Belief state reconciliation protocol (floating-point divergence) |
| Low | **Medical imaging domain model** — registered image prior |
| Low | Formal convergence rate analysis as function of covariance spectrum |

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

*NRT — March 2026*
