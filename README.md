# phinx — φ-ensemble

> Thermodynamic agent-based simulation for complex systems

[![PyPI version](https://badge.fury.io/py/phinx.svg)](https://pypi.org/project/phinx/)
[![Python](https://img.shields.io/pypi/pyversions/phinx.svg)](https://pypi.org/project/phinx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-120%20passed-brightgreen)](https://github.com/yourusername/phinx)

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

**phinx integrates all of them** into one pipeline with a unified survival function Φ, computable in real time on local hardware.

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

---

## Mathematical Foundation

The unified survival function:

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
Micro  (Agent ψᵢ)    →  s, π, P(H), ε, E
                              ↓
Ensemble (Z, T*, S, F) →  thermodynamic interface
                              ↓
Macro  (Ψ, D, ⟨O⟩)   →  emergent patterns
                              ↑ (feedback)
```

---

## Installation

```bash
pip install phinx
```

With optional dependencies:

```bash
pip install phinx[fast]    # numba JIT acceleration
pip install phinx[output]  # OSC + WebSocket real-time output
pip install phinx[viz]     # matplotlib visualization
pip install phinx[all]     # everything
```

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

### Game Theory — payoff matrices + ESS

```python
from phinx import (
    PRISONERS_DILEMMA, STAG_HUNT, HARMONY,
    is_ess, population_dynamics, pareto_efficiency
)

# Check if full cooperation is ESS
print(is_ess(1.0, HARMONY))      # True
print(is_ess(1.0, PRISONERS_DILEMMA))  # False

# Replicator dynamics simulation
result = population_dynamics(PRISONERS_DILEMMA, initial_coop=0.5, steps=200)
print(f"Converged to cooperation rate: {result['converged_to']:.3f}")

# Pareto efficiency
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

# Both channels at once
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

# Multi-scale analysis
state = grid.state_matrix()
analysis = fractal_dim_multiscale(state)
print(f"D(3-scale)={analysis['D_3scale']:.3f}  "
      f"healthy={analysis['is_healthy']}")

# Phase transition detection from history
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
# Minimal installation loop
import phinx
import numpy as np
from phinx.output.realtime import RealtimeOutput

grid     = phinx.EnsembleGrid(N=32)
ensemble = phinx.ThermoEnsemble(grid, M=64)
output   = RealtimeOutput(osc_target=("127.0.0.1", 9000))

loop = phinx.PhiLoop(grid, ensemble, fps=60)
loop.run(n_frames=3600, callback=output.send)  # 60s at 60fps
```

---

## Performance

Benchmarks on standard hardware (pure Python, no numba):

| Grid size | step() | compute_phi() | Total/frame |
|---|---|---|---|
| N=8 | ~1ms | ~2ms | ~3ms ✓ |
| N=16 | ~5ms | ~3ms | ~8ms ✓ |
| N=32 | ~20ms | ~4ms | ~24ms ✓ |

Install `phinx[fast]` for numba JIT acceleration (10–50× speedup on the grid loop).

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

**이채문 (Lee Chae-moon)**  
Research Director, EGUN GCC Co., Ltd.  
AI Specialist Instructor, Gyeonggi Province  
NEMAF 2025 — Interactive Media Art

GitHub: [@yourusername](https://github.com/yourusername)
