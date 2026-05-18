# Changelog

## [0.1.0] - 2025-05-18

### Added
- `phinx.core.agent` — Agent state vector ψᵢ with Bayesian update + raindrop collision
- `phinx.core.collision` — Pairwise collision probability P = exp(−d/r₀)
- `phinx.grid.automata` — N×N EnsembleGrid cellular automata
- `phinx.grid.fractal` — Box-counting fractal dimension D
- `phinx.game.payoff` — PayoffMatrix, ESS detection, replicator dynamics
- `phinx.thermo.ensemble` — ThermoEnsemble: Z, T*, S, F, phase transition detection
- `phinx.phi` — Unified survival function Φ + PhiLoop real-time runner
- `phinx.output.realtime` — OSC, WebSocket, Console output channels
- 120 tests, all passing
