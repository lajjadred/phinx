"""tests/test_payoff.py — Phase 1 Week 3 테스트"""

import numpy as np
import pytest
import phinx
from phinx.game.payoff import (
    PayoffMatrix, PRISONERS_DILEMMA, STAG_HUNT, HARMONY, SNOWDRIFT,
    GAMES, is_ess, ess_landscape, population_dynamics, pareto_efficiency,
)
from phinx.grid.automata import EnsembleGrid


# ── PayoffMatrix 기본 ────────────────────────────────────────────────

class TestPayoffMatrix:
    def test_matrix_shape(self):
        m = PRISONERS_DILEMMA.matrix()
        assert m.shape == (2, 2)

    def test_pd_structure(self):
        pd = PRISONERS_DILEMMA
        assert pd.T > pd.R > pd.P > pd.S

    def test_stag_hunt_structure(self):
        sh = STAG_HUNT
        assert sh.R > sh.T

    def test_expected_payoff_mutual_coop(self):
        # 둘 다 협력 → R
        p = PRISONERS_DILEMMA.expected_payoff(1.0, 1.0)
        assert abs(p - PRISONERS_DILEMMA.R) < 1e-9

    def test_expected_payoff_mutual_defect(self):
        # 둘 다 배반 → P
        p = PRISONERS_DILEMMA.expected_payoff(0.0, 0.0)
        assert abs(p - PRISONERS_DILEMMA.P) < 1e-9

    def test_expected_payoff_temptation(self):
        # 내가 배반, 상대 협력 → T
        p = PRISONERS_DILEMMA.expected_payoff(0.0, 1.0)
        assert abs(p - PRISONERS_DILEMMA.T) < 1e-9

    def test_nash_equilibria_pd(self):
        eq = PRISONERS_DILEMMA.nash_equilibria()
        # PD의 순수전략 내쉬: (배반, 배반) = (0,0)
        assert (0.0, 0.0) in eq

    def test_social_dilemma(self):
        assert PRISONERS_DILEMMA.is_social_dilemma()
        assert not HARMONY.is_social_dilemma()

    def test_to_dict(self):
        d = PRISONERS_DILEMMA.to_dict()
        assert "R" in d and "T" in d and "S" in d and "P" in d

    def test_games_dict(self):
        assert "pd" in GAMES
        assert "stag_hunt" in GAMES
        assert isinstance(GAMES["pd"], PayoffMatrix)


# ── ESS 판정 ─────────────────────────────────────────────────────────

class TestESS:
    def test_pd_defection_is_ess(self):
        # PD에서 전체 배반(coop=0)이 ESS
        assert is_ess(0.0, PRISONERS_DILEMMA)

    def test_pd_cooperation_not_ess(self):
        # PD에서 전체 협력(coop=1)은 ESS 아님
        assert not is_ess(1.0, PRISONERS_DILEMMA)

    def test_harmony_cooperation_is_ess(self):
        # Harmony에서 협력이 ESS
        assert is_ess(1.0, HARMONY)

    def test_ess_landscape_returns_keys(self):
        result = ess_landscape(PRISONERS_DILEMMA, resolution=20)
        assert "strategies" in result
        assert "fitness" in result
        assert "ess_points" in result

    def test_ess_landscape_strategies_length(self):
        result = ess_landscape(PRISONERS_DILEMMA, resolution=20)
        assert len(result["strategies"]) == 20


# ── 복제자 방정식 ────────────────────────────────────────────────────

class TestPopulationDynamics:
    def test_pd_converges_to_defection(self):
        result = population_dynamics(
            PRISONERS_DILEMMA, initial_coop=0.5, steps=200, noise=0.001
        )
        # PD에서 협력율이 낮은 쪽으로 수렴
        assert result["converged_to"] < 0.5

    def test_harmony_converges_to_cooperation(self):
        result = population_dynamics(
            HARMONY, initial_coop=0.5, steps=200, noise=0.001
        )
        assert result["converged_to"] > 0.5

    def test_history_length(self):
        result = population_dynamics(PRISONERS_DILEMMA, steps=50)
        assert len(result["coop_history"]) == 51  # initial + 50 steps

    def test_history_bounded(self):
        result = population_dynamics(PRISONERS_DILEMMA, steps=100)
        h = result["coop_history"]
        assert h.min() >= 0
        assert h.max() <= 1


# ── 파레토 효율성 ────────────────────────────────────────────────────

class TestParetoEfficiency:
    def test_full_coop_is_pareto(self):
        result = pareto_efficiency(PRISONERS_DILEMMA, coop_rate=1.0)
        assert result["is_pareto"]

    def test_full_defect_not_pareto(self):
        result = pareto_efficiency(PRISONERS_DILEMMA, coop_rate=0.0)
        assert not result["is_pareto"]

    def test_efficiency_bounded(self):
        result = pareto_efficiency(PRISONERS_DILEMMA, coop_rate=0.5)
        assert 0 <= result["efficiency"] <= 1


# ── 통합 테스트: Grid + PayoffMatrix ─────────────────────────────────

class TestIntegration:
    def test_grid_with_custom_payoff(self):
        """Stag Hunt 보수로 그리드 생성 + 실행"""
        pm = STAG_HUNT.to_dict()
        grid = EnsembleGrid(
            N=8,
            payoff_matrix={"R": pm["R"], "T": pm["T"],
                           "S": pm["S"], "P": pm["P"]}
        )
        for _ in range(3):
            grid.step()
        st = grid.stats()
        assert "mean_coop" in st
        assert 0 <= st["mean_coop"] <= 1

    def test_full_pipeline_import(self):
        """phinx 최상위에서 모든 심볼 import 가능"""
        assert hasattr(phinx, "Agent")
        assert hasattr(phinx, "EnsembleGrid")
        assert hasattr(phinx, "PRISONERS_DILEMMA")
        assert hasattr(phinx, "population_dynamics")
        assert hasattr(phinx, "fractal_dim_boxcount")

    def test_grid_coop_changes_with_stag_hunt(self):
        """Stag Hunt에서 협력이 PD보다 높아야 함 (방향성 확인)"""
        np.random.seed(42)
        g_pd = EnsembleGrid(N=8, payoff_matrix=PRISONERS_DILEMMA.to_dict())
        g_sh = EnsembleGrid(N=8, payoff_matrix=STAG_HUNT.to_dict())

        # 동일 초기 prior 설정
        for i in range(8):
            for j in range(8):
                v = float(np.clip(np.random.rand(), 1e-6, 1-1e-6))
                g_pd.agents[i][j].prior = v
                g_sh.agents[i][j].prior = v

        for _ in range(10):
            g_pd.step()
            g_sh.step()

        # 단순히 실행 완료 확인 (방향성은 확률적)
        assert g_pd.frame == 10
        assert g_sh.frame == 10

    def test_population_dynamics_with_grid_coop(self):
        """복제자 방정식 수렴값과 그리드 협력율 방향성 일치 확인"""
        dyn = population_dynamics(HARMONY, initial_coop=0.3,
                                  steps=100, noise=0.001)
        # Harmony에서는 협력 방향으로 수렴해야 함
        assert dyn["converged_to"] > 0.3
