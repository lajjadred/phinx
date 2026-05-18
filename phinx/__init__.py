"""
phinx — φ-ensemble
===================
Thermodynamic agent-based simulation for complex systems.

셀룰러 오토마타 + 베이즈 갱신 + 게임이론 + 열역학 앙상블 + 실시간 출력
통합 파이프라인 패키지.

Quick Start
-----------
>>> from phinx import Agent
>>> a = Agent(prior=0.6)
>>> b = Agent(prior=0.3)
>>> a.meet(b, distance=0.5)
>>> print(a.act())
"""

__version__ = "0.1.0"
__author__ = "이채문"

from phinx.core.agent import Agent, PAYOFF_PD, PAYOFF_SH, PAYOFF_HG
from phinx.core.collision import (
    collision_prob,
    ensemble_collision_prob,
    process_collisions,
)
from phinx.grid.automata import EnsembleGrid
from phinx.grid.fractal import (
    fractal_dim_boxcount,
    fractal_dim_multiscale,
    fractal_dim_history,
)
from phinx.output.realtime import RealtimeOutput, ConsoleOutput, OSCOutput

from phinx.thermo.ensemble import ThermoEnsemble
from phinx.phi import compute_phi, PhiLoop

from phinx.game.payoff import (
    PayoffMatrix,
    PRISONERS_DILEMMA,
    STAG_HUNT,
    HARMONY,
    SNOWDRIFT,
    GAMES,
    is_ess,
    ess_landscape,
    population_dynamics,
    pareto_efficiency,
)

__all__ = [
    "Agent", "PAYOFF_PD", "PAYOFF_SH", "PAYOFF_HG",
    "collision_prob", "ensemble_collision_prob", "process_collisions",
    "EnsembleGrid",
    "fractal_dim_boxcount", "fractal_dim_multiscale", "fractal_dim_history",
    "RealtimeOutput", "ConsoleOutput", "OSCOutput",
    "ThermoEnsemble", "compute_phi", "PhiLoop", "PRISONERS_DILEMMA", "STAG_HUNT", "HARMONY",
    "SNOWDRIFT", "GAMES", "is_ess", "ess_landscape",
    "population_dynamics", "pareto_efficiency",
]
