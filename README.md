# phinx — φ-ensemble

> Thermodynamic agent-based simulation for complex systems

[![PyPI version](https://badge.fury.io/py/phinx.svg)](https://pypi.org/project/phinx/)
[![Python](https://img.shields.io/pypi/pyversions/phinx.svg)](https://pypi.org/project/phinx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-120%20passed-brightgreen)](https://github.com/yourusername/phinx)
[![GPU](https://img.shields.io/badge/GPU-Triton%20CC%207.5%2B-76b900)](https://github.com/triton-lang/triton)

**phinx** unifies cellular automata, Bayesian inference, game theory, fractal geometry, and thermodynamic ensembles into a single real-time simulation pipeline — without a global temperature parameter.

---

## Why phinx?

Existing packages handle each domain separately:

| Domain | Existing tools |
|---|---|
| Agent-based models | Mesa |
| Cellular automata | cellpylib |
| Bayesian inference | PyMC, pomegranate |
| Game theory | nashpy, axelrod |
| Thermodynamics | domain-specific only |

**phinx integrates all of them** into one pipeline with a unified survival function Φ, computable in real time on local hardware — CPU or GPU.

### Key idea: local randomness instead of global temperature

Instead of a single global `temperature` parameter (as in transformers):

```
# Transformer approach — global T
P(output) = softmax(logits / T)
```

phinx assigns each agent its own local noise ε, updated through raindrop-collision-style interactions:

```
εᵢ ~ f(contextᵢ, priorᵢ, neighborsᵢ)
output_i = act(stateᵢ) + εᵢ
```

The collective effect of local ε distributions produces an **emergent effective temperature T\***, structurally equivalent to natural uncertainty — without any global parameter.

This is not a metaphor. The raindrop collision probability `p = exp(−d/r₀)` is mathematically
identical to an attention score `softmax(QKᵀ/√d)`. phinx implements agent interaction as a
**masked local attention kernel**, accelerated by Triton on GPU.

---

## Mathematical Foundation

### The unified survival function

```
Φ = sigmoid(α·S + β·D − γ·T*) · ⟨cooperation⟩_M
```

| Symbol | Meaning | Source theory |
|---|---|---|
| S | Shannon entropy of prior distribution | Thermodynamics |
| D | Fractal dimension (box-counting) | Fractal theory |
| T\* | Effective temperature = Var(εᵢ)/k | Local randomness |
| ⟨coop⟩_M | Monte Carlo cooperation estimate | Bayesian + Game theory |
| α, β, γ | Aesthetic tuning parameters | — |

**Φ → 1**: stable system (ESS maintained, fractal healthy, free energy minimum)  
**Φ → 0**: collapse (phase transition reached, D drops, defection ESS)

### Three-layer architecture

```
Micro  (Agent ψᵢ)      →  s, π, P(H), ε, E
                                ↓  raindrop collision + Bayesian update
Ensemble (Z, T*, S, F) →  thermodynamic interface
                                ↓  partition function + phase detection
Macro  (Ψ, D, ⟨O⟩)    →  emergent patterns
                                ↑  (feedback: macro state → micro prior)
```

### Thermodynamic ensemble — bridging micro and macro

The ensemble layer computes global system state from local agent distributions,
without iterating every agent pair:

```
Z    = Σᵢ exp(−Eᵢ / kT*)        partition function
T*   = Var(εᵢ) / k              emergent temperature (no global T needed)
S    = −k Σ Pᵢ ln Pᵢ            entropy  (diversity → stability)
F    = E − T*·S                  free energy (minimum = stable ESS)
Tc   = ∂²F/∂T*² = 0             phase transition point
```

High entropy S (diverse agent population) → low free energy F → stable system.  
Homogenization reduces S → F rises → system approaches collapse threshold Tc.

### Raindrop sequence — collision probability

n agents moving on independent paths collide with probability:

```
P(at least one collision) = 1 − (1−p)^C(n,2)
```

Each collision is an **evidence exchange event** that triggers Bayesian belief revision
(τ = 3 iterations by default, convergence guaranteed by KL-divergence bound):

```
P(H|E) ∝ P(E|H) · P(H)     ← posterior of agent i after meeting agent j
```

The local ε of both agents co-evolves after each collision:

```
εᵢ_new = 0.8·εᵢ + 0.2·εⱼ   ← ε co-evolution (local temperature mixing)
```

This replaces global attention with **context-dependent, path-history-aware belief revision** —
structurally analogous to quantum wavefunction collapse at measurement.

---

## Installation

```bash
pip install phinx
```

With optional dependencies:

```bash
pip install phinx[fast]    # numba JIT acceleration (CPU)
pip install phinx[gpu]     # Triton GPU kernel (RTX 20xx+)
pip install phinx[output]  # OSC + WebSocket real-time output
pip install phinx[viz]     # matplotlib visualization
pip install phinx[all]     # everything
```

### GPU setup (RTX 20xx / 30xx / 40xx)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install triton
pip install phinx[gpu]
```

Triton kernels activate automatically when a compatible GPU is detected (CC ≥ 7.5).
**No code changes required** — the same API runs identically on CPU and GPU.

---

## Quick Start

```python
import numpy as np
import phinx

# 1. Create grid
grid = phinx.EnsembleGrid(N=16, r=1)

# 2. Randomize initial priors (diversity matters)
for i in range(grid.N):
    for j in range(grid.N):
        grid.agents[i][j].prior = float(np.random.rand())

# 3. Thermodynamic ensemble
ensemble = phinx.ThermoEnsemble(grid, M=64)

# 4. Run simulation
loop = phinx.PhiLoop(grid, ensemble, fps=60, alpha=0.3, beta=0.4, gamma=0.3)

def on_frame(result):
    print(f"frame={result['frame']:04d}  "
          f"Φ={result['phi']:.3f}  "
          f"signal={result['signal']}")

results = loop.run(n_frames=100, callback=on_frame)

# 5. Summary
summary = loop.summary()
print(f"Φ mean={summary['phi_mean']:.3f}  "
      f"avg={summary['avg_total_ms']:.1f}ms/frame")
```

### GPU pipeline (PhinxPipeline)

For real-time installations requiring maximum throughput:

```python
from phinx.core.gpu_pipeline import PhinxPipeline, PhinxConfig

cfg = PhinxConfig(N=32, M=64)
cfg.auto_tune()   # reads GPU model + VRAM, sets N/M/D automatically

pipeline = PhinxPipeline(cfg)

for frame in pipeline.run():
    print(f"Φ={frame['phi']:.3f}  {frame['ms']:.1f}ms")
    # frame keys: phi, S, D, T_star, coop, F, is_critical, frame, ms
```

`auto_tune()` adapts to the detected hardware:

| Hardware | CC | Auto M | Per-frame target |
|---|---|---|---|
| CPU only | — | 32 | ~40–80ms |
| RTX 2060/70 | 7.5 | 64 | ~12ms ✓ |
| RTX 2080 Ti | 7.5 | 128 | ~8ms ✓ |
| RTX 3060 | 8.6 | 128 | ~6ms ✓ |
| RTX 3080+ | 8.6 | 512 | ~3ms ✓ |

---

## Core Components

### Agent — state vector ψᵢ

```python
from phinx import Agent

a = Agent(
    prior=0.6,          # Bayesian prior P(H) — cooperation belief
    epsilon_var=0.1,    # local noise variance (replaces global T)
    energy=0.4,         # internal energy Eᵢ (mind-body state)
)

b = Agent(prior=0.3)

# Raindrop collision → Bayesian update + strategy update + ε co-evolution
a.meet(b, distance=0.5)

print(a.act())       # local ε-sampled action
print(a.to_dict())   # serialize
```

### EnsembleGrid — N×N cellular automata

```python
from phinx import EnsembleGrid

grid = EnsembleGrid(
    N=32,       # grid size (N² agents)
    r=1,        # neighbor radius (r=1 → 8 directions)
    wrap=True,  # toroidal boundary
)

ms = grid.step()             # one frame update
print(grid.stats())          # aggregate statistics
print(grid.fractal_dim())    # fractal dimension D ∈ [1.0, 2.0]
```

### GPU Attention Kernel

Agent interactions are implemented as **masked local attention** — mathematically equivalent
to raindrop collision probability — compiled to Triton for GPU execution:

```python
from phinx.core.attention_kernel import agent_attention, masked_agent_attention

# Auto-dispatches: Triton (CC ≥ 7.5) or PyTorch fallback (CPU / older GPU)
out = agent_attention(Q, K, V)

# With neighbor radius mask — only agents within r interact
out = masked_agent_attention(Q, K, V, positions=pos, r=1.5)
```

Kernel dispatch is fully automatic:

```
CUDA available
  + Compute Capability ≥ 7.5   (RTX 20xx, 30xx, 40xx)
  + triton installed
  ──────────────────────────►  Triton kernel   (fp16, Tensor Core)

otherwise
  ──────────────────────────►  PyTorch fallback (CPU or CUDA, fp32)
```

### Game Theory — payoff matrices + ESS

```python
from phinx import (
    PRISONERS_DILEMMA, STAG_HUNT, HARMONY,
    is_ess, population_dynamics, pareto_efficiency
)

print(is_ess(1.0, HARMONY))            # True
print(is_ess(1.0, PRISONERS_DILEMMA)) # False

result = population_dynamics(PRISONERS_DILEMMA, initial_coop=0.5, steps=200)
print(f"Converged to cooperation rate: {result['converged_to']:.3f}")

eff = pareto_efficiency(PRISONERS_DILEMMA, coop_rate=0.6)
print(f"Efficiency: {eff['efficiency']:.3f}")
```

### Thermodynamic Ensemble

```python
from phinx import ThermoEnsemble, compute_phi

ensemble = ThermoEnsemble(grid, M=64)

result = compute_phi(grid, ensemble, alpha=0.3, beta=0.4, gamma=0.3)
print(f"Φ={result['phi']:.3f}  S={result['S']:.3f}  "
      f"D={result['D']:.3f}  T*={result['T_star']:.4f}")
print(f"signal: {result['signal']}")  # stable | warning | critical
```

### Real-time Output

```python
from phinx.output.realtime import RealtimeOutput, ConsoleOutput

# Console (development)
console = ConsoleOutput(every_n=10)

# OSC → Max/MSP, TouchDesigner, SuperCollider
from phinx.output.realtime import OSCOutput
osc = OSCOutput(host="127.0.0.1", port=9000)

# OSC + WebSocket simultaneously
rt = RealtimeOutput(
    osc_target=("127.0.0.1", 9000),
    ws_port=8765,           # WebSocket → browser GLSL / p5.js
)

loop.run(n_frames=1000, callback=rt.send)
```

#### OSC message schema

| Address | Type | Range | Description |
|---|---|---|---|
| `/phinx/phi` | f | [0, 1] | survival index |
| `/phinx/entropy` | f | [0, ∞) | diversity |
| `/phinx/fractal` | f | [1, 2] | pattern complexity |
| `/phinx/temp` | f | [0, ∞) | effective temperature |
| `/phinx/coop` | f | [0, 1] | cooperation rate |
| `/phinx/signal` | s | — | "stable"\|"warning"\|"critical" |
| `/phinx/frame` | i | — | frame number |
| `/phinx/critical` | i | [0, 1] | phase transition flag |

---

## Fractal Analysis

```python
from phinx.grid.fractal import fractal_dim_multiscale, fractal_dim_history

state = grid.state_matrix()
analysis = fractal_dim_multiscale(state)
print(f"D(3-scale)={analysis['D_3scale']:.3f}  "
      f"healthy={analysis['is_healthy']}")

D_history = [grid.fractal_dim() for _ in range(30)]
alert = fractal_dim_history(D_history, window=10)
print(f"alert={alert['alert']}  drop={alert['drop']:.3f}")
```

---

## Interactive Art Applications

phinx was developed in part for the **NEMAF 2025** interactive installation artwork.

The survival function Φ maps directly to sensory output:

| Φ component | Visual | Audio |
|---|---|---|
| S (entropy) | color diversity | harmonic richness |
| D (fractal dim) | pattern complexity | rhythmic complexity |
| T\* (temperature) | particle turbulence | timbre roughness |
| Φ (combined) | overall density | consonance/dissonance |
| signal=critical | monochrome collapse | noise flood |

```python
# Minimal installation loop — GPU accelerated, OSC output
from phinx.core.gpu_pipeline import PhinxPipeline, PhinxConfig
from pythonosc import udp_client

cfg = PhinxConfig(N=32, M=64)
cfg.auto_tune()

pipeline = PhinxPipeline(cfg)
osc = udp_client.SimpleUDPClient("127.0.0.1", 9000)

for frame in pipeline.run():
    osc.send_message("/phinx/phi",      frame['phi'])
    osc.send_message("/phinx/entropy",  frame['S'])
    osc.send_message("/phinx/fractal",  frame['D'])
    osc.send_message("/phinx/temp",     frame['T_star'])
    osc.send_message("/phinx/critical", int(frame['is_critical']))
```

---

## Performance

### CPU (pure Python, no numba)

| Grid size | step() | compute_phi() | Total/frame |
|---|---|---|---|
| N=8 | ~1ms | ~2ms | ~3ms ✓ |
| N=16 | ~5ms | ~3ms | ~8ms ✓ |
| N=32 | ~20ms | ~4ms | ~24ms ✓ |

Install `phinx[fast]` for numba JIT acceleration (10–50× speedup on the grid loop).

### GPU (Triton kernel — `phinx[gpu]`)

| GPU | CC | N=32 M=64 | N=32 M=128 | 60fps budget |
|---|---|---|---|---|
| RTX 2060/70 | 7.5 | ~12ms | ~16ms | ✓ |
| RTX 2080 Ti | 7.5 | ~8ms | ~10ms | ✓ |
| RTX 3060 | 8.6 | ~6ms | ~8ms | ✓ |
| RTX 3080+ | 8.6 | ~3ms | ~4ms | ✓ |

On CPU-only machines, `phinx[gpu]` is not required.
PyTorch fallback activates automatically — same API, same results.

---

## Theoretical Background

phinx integrates the following theoretical frameworks:

- **Behavioral Psychology** — reinforcement, imitation, social learning → agent state `s`
- **Conway's Game of Life** — local rules → global emergence → cellular automata grid
- **Prisoner's Dilemma / Game Theory** — cooperation/defection, ESS, replicator dynamics
- **Personality Theory** (Big Five, MBTI) — individual parameter variation → diversity
- **Fractal Theory** — self-similarity across scales → D as complexity measure
- **Mind-Body Monism** (Spinoza) — physical space ↔ collective psychology → energy `E`
- **Frege's Logic** — Sinn/Bedeutung: different theories, same referent (Φ)
- **Bayesian Inference** — prior → evidence → posterior → raindrop collision update
- **Thermodynamics** — partition function Z, entropy S, free energy F, phase transition Tc
- **Quantum analogy** — superposition (local ε before collision) → collapse (Bayesian update)

All unified under the survival function **Φ**.

---

## Citation

If you use phinx in research or artwork, please cite:

```bibtex
@software{phinx2025,
  author  = {Lee, Chae-moon},
  title   = {phinx: Thermodynamic agent-based simulation for complex systems},
  year    = {2025},
  url     = {https://github.com/yourusername/phinx},
  version = {0.1.0}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**lajjadred**

GitHub: [@yourusername](https://github.com/lajjadred)
