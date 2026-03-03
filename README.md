# NRT

**Null Refinement Transport**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-13%20passing-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)]()

> *Project the transmission into the null space of the receiver's knowledge, and send only what survives the projection.*

---

## Table of contents

1. [Overview](#overview)
2. [Problem statement](#problem-statement)
3. [Mathematical foundation](#mathematical-foundation)
4. [The belief covariance — what makes this different](#the-belief-covariance)
5. [Protocol design](#protocol-design)
6. [Prior knowledge and acceleration](#prior-knowledge-and-acceleration)
7. [Loss resilience](#loss-resilience)
8. [Empirical results](#empirical-results)
9. [Limitations](#limitations)
10. [Installation](#installation)
11. [Usage](#usage)
12. [Project structure](#project-structure)
13. [Roadmap](#roadmap)

---

## Overview

Current transport protocols transmit data as if the receiver knows nothing. The sender shouts the answer into a wire without asking: *what does the other side already believe?*

NRT (Null Refinement Transport) starts from a different primitive. The fundamental unit of transfer is not a byte — it is a **refinement**: a correction operator that, when applied to the receiver's current belief state, collapses its uncertainty along the most informative axis available.

The mechanism: the receiver maintains a belief covariance $\Sigma_r$ — a formal description of what it is uncertain about, and how uncertain. The sender reads this covariance, projects the correction into its eigenbasis, and transmits only the components that address genuine uncertainty. Dimensions where the receiver is already confident receive no transmission at all.

When a correction is lost in transit, the sender does not retransmit a copy. It recomputes the correction against the receiver's *current* belief — which has evolved since the original packet was sent. The retransmission is smaller, more targeted, and never a repeat.

Transfer terminates when the receiver's uncertainty falls below $\varepsilon$ — a threshold declared by the application, not by the protocol.

---

## Problem statement

The Shannon limit defines the maximum reliable bit rate for a channel. It does not define *which bits to send*. Classical protocols transmit all bits of the data, in order. This is optimal only when the receiver has zero prior knowledge.

In practice, receivers often hold partial knowledge:

- A server receiving a model update already has the previous checkpoint.
- A vehicle receiving a sensor fusion update holds its own local sensor readings.
- A replica receiving a database sync already has a previous snapshot.
- A device receiving a software update already has the running version.

In every such case, the minimum-cost transmission is the information the receiver does not already have — the divergence between the receiver's current belief and the truth, expressed in the basis of the receiver's maximum uncertainty.

> [!NOTE]
> NRT does not compress data. It transmits *what the receiver doesn't know*, in the order the receiver needs it most. With no prior knowledge, it behaves identically to progressive transfer. With a good prior, it may transmit nothing at all.

---

## Mathematical foundation

### Belief state

Let $\mathbf{X} \in \mathbb{R}^{m \times n}$ be the sender's data. The receiver maintains:

- An estimate $\hat{\mathbf{X}}^{(t)} \in \mathbb{R}^{m \times n}$ — its current best reconstruction
- A belief covariance $\Sigma_r$ — a description of its uncertainty, factored as $\Sigma_L \otimes \Sigma_R$ (Kronecker structure over left and right matrix directions)

The **residual** at round $t$ is:

$$\mathbf{R}^{(t)} = \mathbf{X} - \hat{\mathbf{X}}^{(t)}$$

### Uncertainty eigenbasis

The belief covariance factorises as:

$$\Sigma_L = \mathbf{V}_L \mathbf{\Lambda}_L \mathbf{V}_{L}^\top, \qquad \Sigma_R = \mathbf{V}_R \mathbf{\Lambda}_R \mathbf{V}_{R}^\top$$

The combined uncertainty over outer-product directions is:

$$\lambda_{ij} = \lambda_{i}^{L} \cdot \lambda_{j}^{R}$$

A large $\lambda_{ij}$ means the receiver is highly uncertain about the direction $\mathbf{u}_i \mathbf{v}_{j}^\top$. A $\lambda_{ij} = 0$ means the receiver *already knows* that direction — it is never transmitted.

### Correction projection

The scalar correction for direction $(i, j)$ is:

$$c_{ij} = \mathbf{u}_{i}^\top \mathbf{R}^{(t)} \mathbf{v}_j$$

NRT transmits the top-$k$ directions sorted by $\lambda_{ij}$ (highest uncertainty first), carrying their corrections $c_{ij}$.

> [!IMPORTANT]
> This is the critical distinction from SVD-based approaches. SVD of the residual prioritises directions of highest **variance in the data**. NRT prioritises directions of highest **uncertainty in the receiver**. These coincide only when $\Sigma_r = \mathbf{I}$ (no prior). When the receiver has a prior, they diverge — and NRT becomes strictly more efficient.

### Belief update

The receiver applies each received correction:

$$\hat{\mathbf{X}}^{(t+1)} = \hat{\mathbf{X}}^{(t)} + \sum_{\text{received}} c_{ij} \cdot \mathbf{u}_i \mathbf{v}_{j}^\top$$

After applying a correction for direction $(i,j)$, the receiver projects it out of the covariance:

$$\Sigma_L \leftarrow \Sigma_L - \lambda_{i}^{L} \cdot \mathbf{u}_i \mathbf{u}_{i}^\top, \qquad \Sigma_R \leftarrow \Sigma_R - \lambda_{j}^{R} \cdot \mathbf{v}_j \mathbf{v}_{j}^\top$$

That direction's uncertainty drops to zero. It will never be transmitted again.

### $\varepsilon$-sufficiency

Transfer terminates when:

$$\frac{\lVert \mathbf{X} - \hat{\mathbf{X}}^{(t)} \rVert_F}{\lVert \mathbf{X} \rVert_F} < \varepsilon$$

$\varepsilon$ is declared by the application layer. $\varepsilon = 0$ requires exact reconstruction. $\varepsilon = 0.05$ means 5% relative error is acceptable. The protocol terminates as soon as the threshold is met — no unnecessary transmission.

### Retransmission norm bound

When a packet is lost, the sender recomputes the correction against the receiver's current belief $\hat{\mathbf{X}}^{(t')}$, where $t' > t$ (other corrections have since been applied). Since convergence is monotone:

$$\lVert \mathbf{R}^{(t')} \rVert_F \leq \lVert \mathbf{R}^{(t)} \rVert_F \implies \lVert \text{retransmit} \rVert_F \leq \lVert \text{original} \rVert_F$$

The retransmission is always at most as large as the original, and in practice smaller. It is never a copy.

---

## The belief covariance

The covariance $\Sigma_r$ is what gives NRT its character. It is not just a compression tool — it is the receiver's formal declaration of what it doesn't know.

**Zero prior** — receiver has no knowledge:

$$\Sigma_L = \mathbf{I}_m, \quad \Sigma_R = \mathbf{I}_n$$

All directions are equally uncertain. NRT degenerates to transmitting corrections in order of magnitude — equivalent to residual SVD. This is the correct degenerate case: NRT is never worse than progressive transfer.

**Spectral prior** — receiver has the top-$p$ SVD components of $\mathbf{X}$:

$$\Sigma_L = \mathbf{I}_m - \mathbf{U}_p \mathbf{U}_{p}^\top, \quad \Sigma_R = \mathbf{I}_n - \mathbf{V}_p \mathbf{V}_{p}^\top$$

Known directions have $\lambda = 0$ — they are explicitly excluded from the eigenbasis and never transmitted. The sender only corrects what the receiver genuinely does not know. If the prior is good enough to satisfy $\varepsilon$ already, **zero bytes are transmitted**.

---

## Protocol design

### Session phases

```
Sender                                Receiver
  │                                      │
  │──── HELLO (epsilon, shape) ────────► │
  │◄─── SEED (Σ_L, Σ_R, X̂ hash) ──────  │  receiver sends its covariance
  │                                      │
  │  [check: is residual already < ε?]   │
  │  [if yes: send DONE, 0 bytes sent]   │
  │                                      │
  │──── REFINE (round=1, frames) ──────► │  top-k directions by λ_ij
  │◄─── ACK (residual_norm, stop?) ───── │
  │                                      │
  │──── REFINE (round=2, frames) ──────► │
  │◄─── STOP (residual < ε) ───────────  │
  │                                      │
  │  (closes connection)                 │
```

### Refinement frame

```
[round_idx  : uint32]   belief epoch this correction targets
[u_i        : float32[] (m,)]   left eigenvector of Σ_L
[v_j        : float32[] (n,)]   right eigenvector of Σ_R
[c_ij       : float32]          scalar correction u_i^T R v_j
[lambda_ij  : float32]          uncertainty of this direction at send time
[is_final   : uint8]            1 if sender believes ε is satisfied
```

### ACK frame

```
[round_idx    : uint32]
[residual     : float32]   receiver's current relative error
[stop         : uint8]     1 = sufficient, 0 = continue
```

---

## Prior knowledge and acceleration

| Prior type | $\Sigma_r$ structure | Effect on transmission |
|:-----------|:---------------------|:-----------------------|
| Zero (no knowledge) | $\mathbf{I}_m \otimes \mathbf{I}_n$ — uniform | Baseline: all directions equally uncertain, ordered by correction magnitude |
| Spectral (receiver has top-$p$ SVD) | Known directions projected out: $\lambda_{ij} = 0$ | **Accelerates**: known directions skipped entirely; may converge in 0 rounds |
| Wrong/noisy prior | Covariance poorly calibrated | **Degrades**: if $\Sigma_r$ does not reflect actual uncertainty, sender may transmit in wrong order |

> [!WARNING]
> The prior must accurately represent the receiver's actual uncertainty. A miscalibrated covariance causes the sender to prioritise the wrong directions. Unlike SVD-based approaches, NRT's performance is sensitive to covariance quality — not just data similarity.

---

## Loss resilience

Under packet loss, NRT does not stall or retransmit copies. The sender tracks the receiver's last acknowledged belief state and recomputes the correction against that state. Because the covariance updates as other corrections arrive, the retransmission targets the receiver's *current* uncertainty — which differs from when the original was sent.

Formally: the lost correction targeted directions with uncertainty $\lambda_{ij}^{(t)}$. If other corrections have since reduced the residual, the retransmission carries smaller $c_{ij}$ values (the residual is smaller) while targeting the same high-uncertainty directions (their $\lambda_{ij}$ is unchanged — never received). The norm bound holds: $\lVert\text{retransmit}\rVert \leq \lVert\text{original}\rVert$.

---

## Empirical results

Benchmark: 32×32 structured matrix (rank-5 signal + noise), 4,096 bytes raw. $\varepsilon = 0.02$, $k = 4$ directions per round.

### Prior comparison

| Prior | Rounds | Bytes sent | % of raw | Converged |
|------:|-------:|-----------:|---------:|:---------:|
| Zero (Σ = I, no knowledge) | 6 | 6,432 | 157.0% | ✓ |
| Spectral, top 10% known | 1 | 1,072 | **26.2%** | ✓ |
| Spectral, top 30% known | 0 | **0** | **0.0%** | ✓ |
| Spectral, top 50% known | 0 | **0** | **0.0%** | ✓ |

The 0-byte cases are correct: the receiver's prior already satisfies $\varepsilon$. No transmission is needed. This is not an edge case — it is the intended behaviour when the receiver is sufficiently informed.

> [!NOTE]
> Zero prior transmits 157% of raw due to per-frame encoding overhead (eigenvectors $\mathbf{u}_i$, $\mathbf{v}_j$ must be transmitted alongside each scalar $c_{ij}$). NRT is not competitive with raw transfer for small matrices with no prior. Its advantage is in large-data, partial-knowledge scenarios.

### Loss resilience

| Channel | Rounds | Converged |
|:--------|-------:|:---------:|
| Clean (0% loss) | 6 | ✓ |
| 10% packet loss | 6 | ✓ |
| 30% packet loss | 6 | ✓ |

Convergence is unaffected by loss rate on this benchmark. Loss increases round count on larger data.

---

## Limitations

> [!CAUTION]
> **NRT is a research prototype (v0.1).** All results use simulated sessions on synthetic data. No real network transport is implemented.

**Covariance overhead.** Each frame must transmit the eigenvectors $\mathbf{u}_i \in \mathbb{R}^{m}$ and $\mathbf{v}_j \in \mathbb{R}^{n}$ alongside the scalar correction $c_{ij}$. For small matrices, this overhead dominates and NRT is less efficient than raw transfer. Efficiency improves as matrix dimensions increase.

**Covariance calibration.** NRT's performance depends on $\Sigma_r$ accurately representing the receiver's uncertainty. A miscalibrated prior causes suboptimal ordering. Unlike SVD-based approaches, which degrade gracefully with a bad prior, NRT with a wrong covariance may transmit less useful information first.

**Computational cost.** Eigendecomposing $\Sigma_L \in \mathbb{R}^{m \times m}$ and $\Sigma_R \in \mathbb{R}^{n \times n}$ per round is $O(m^{3} + n^{3})$. For large weight matrices (e.g. $4096 \times 4096$) this is expensive. Block-diagonal approximations are needed in production.

**Synchronisation requirement.** Both sender and receiver must maintain consistent belief state. Floating-point non-determinism across heterogeneous hardware can cause state divergence. Production use requires deterministic arithmetic or a reconciliation protocol.

**Structured data only.** For data without low-rank structure — encrypted payloads, random bytes — corrections are unstructured and NRT adds overhead with no benefit.

**Prior art.** NRT draws on Wyner–Ziv coding (1976), distributed source coding (Slepian–Wolf 1973), information geometry (Amari 1980s), and semantic communication (active research since ~2019). The contribution is a concrete protocol design: a Kronecker-structured belief covariance, per-direction uncertainty ordering, covariance update rules, and the $\varepsilon$-sufficiency termination mechanism. This is an engineering contribution, not a mathematical discovery.

---

## Installation

```bash
git clone https://github.com/userFRM/nrt.git
cd nrt
pip install -r requirements.txt
```

**Requirements:** `numpy>=1.24`, `scipy>=1.10`, `pytest>=7.0` (tests only)

---

## Usage

```python
import numpy as np
from nrt.session import NRTSession
from nrt.models import zero_prior, spectral_prior
from nrt import NRTConfig

data = np.load("checkpoint.npy")                      # sender's data
prior = spectral_prior(previous_data, keep_fraction=0.20)  # receiver's prior

config = NRTConfig(epsilon=0.02, k_per_round=4)
result = NRTSession().run(data, prior=prior, config=config)

print(f"{result.rounds} rounds, {result.total_bytes} bytes ({result.compression_ratio:.1%} of raw)")
# With a good prior: 0 rounds, 0 bytes — receiver already knew enough
```

### Run tests

```bash
python3 -m pytest tests/ -v
# 13 tests: covariance mechanics, convergence, prior acceleration, retransmit bound
```

---

## Project structure

```
nrt/
├── nrt/
│   ├── __init__.py     NRTConfig, CorrectionFrame, RefinementPacket, SessionResult
│   ├── belief.py       BeliefState — Σ_L⊗Σ_R covariance, eigenbasis, update rules
│   ├── models.py       Prior constructors — zero_prior(), spectral_prior()
│   ├── sender.py       NRTSender — corrections ordered by λ_ij, retransmit()
│   ├── receiver.py     NRTReceiver — apply(), is_sufficient()
│   └── session.py      NRTSession — full simulation with optional packet loss
└── tests/
    ├── test_covariance.py    4 tests — covariance structure, known-direction exclusion, updates
    ├── test_convergence.py   3 tests — monotone convergence, ε-termination, zero-prior degenerate
    ├── test_prior.py         3 tests — spectral prior acceleration, monotone prior improvement
    └── test_retransmit.py    3 tests — norm bound, not-a-copy, convergence under 30% loss
```

---

## Roadmap

| Priority | Item |
|:--------:|:-----|
| High | Real UDP transport — sender reads Σ_r from receiver ACKs |
| High | Benchmark on real data: model checkpoints, file sync, sensor telemetry |
| High | Block-diagonal covariance approximation for large matrices |
| Medium | Belief state reconciliation protocol (floating-point divergence) |
| Medium | Domain-specific covariance models: temporal (Kalman), language model prior |
| Low | Formal convergence rate analysis as function of covariance spectrum |

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

*NRT — March 2026*
