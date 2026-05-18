"""tests/test_grid.py — Phase 1 Week 2 테스트"""

import time
import numpy as np
import pytest

from phinx.grid.automata import EnsembleGrid
from phinx.grid.fractal import (
    fractal_dim_boxcount,
    fractal_dim_multiscale,
    fractal_dim_history,
)


# ── EnsembleGrid 초기화 ──────────────────────────────────────────────

class TestGridInit:
    def test_default_shape(self):
        g = EnsembleGrid(N=8)
        assert len(g.agents) == 8
        assert len(g.agents[0]) == 8

    def test_state_matrix_shape(self):
        g = EnsembleGrid(N=16)
        m = g.state_matrix()
        assert m.shape == (16, 16)

    def test_state_matrix_bounded(self):
        g = EnsembleGrid(N=16)
        m = g.state_matrix()
        assert m.min() >= 0
        assert m.max() <= 1

    def test_repr(self):
        g = EnsembleGrid(N=8)
        r = repr(g)
        assert "EnsembleGrid" in r
        assert "N=8" in r


# ── step() 동작 ─────────────────────────────────────────────────────

class TestGridStep:
    def test_step_runs(self):
        g = EnsembleGrid(N=8)
        elapsed = g.step()
        assert isinstance(elapsed, float)
        assert elapsed >= 0

    def test_frame_increments(self):
        g = EnsembleGrid(N=8)
        assert g.frame == 0
        g.step()
        assert g.frame == 1
        g.step()
        assert g.frame == 2

    def test_state_changes_after_step(self):
        # prior 균일(0.5)이면 베이즈 갱신이 대칭 → 변화 없음 (수학적으로 정상)
        # 랜덤 초기화 후 테스트
        np.random.seed(0)
        g = EnsembleGrid(N=8, wrap=True)
        for i in range(g.N):
            for j in range(g.N):
                g.agents[i][j].prior = float(
                    np.clip(np.random.rand(), 1e-6, 1 - 1e-6)
                )
        before = g.state_matrix().copy()
        for _ in range(5):
            g.step()
        after = g.state_matrix()
        assert not np.allclose(before, after)

    def test_cooperation_matrix_shape(self):
        g = EnsembleGrid(N=8)
        c = g.cooperation_matrix()
        assert c.shape == (8, 8)
        assert c.min() >= 0
        assert c.max() <= 1

    def test_energy_matrix_shape(self):
        g = EnsembleGrid(N=8)
        e = g.energy_matrix()
        assert e.shape == (8, 8)

    def test_epsilon_matrix_shape(self):
        g = EnsembleGrid(N=8)
        eps = g.epsilon_matrix()
        assert eps.shape == (8, 8)
        assert (eps > 0).all()


# ── 성능 벤치마크 ────────────────────────────────────────────────────

class TestGridPerformance:
    def test_step_n16_under_500ms(self):
        g = EnsembleGrid(N=16)
        elapsed = g.step()
        assert elapsed < 500, f"N=16 step: {elapsed:.1f}ms — 너무 느림"

    def test_step_n32_under_2000ms(self):
        g = EnsembleGrid(N=32)
        elapsed = g.step()
        assert elapsed < 2000, f"N=32 step: {elapsed:.1f}ms"

    def test_10_frames_n16(self):
        g = EnsembleGrid(N=16)
        t0 = time.perf_counter()
        for _ in range(10):
            g.step()
        total = (time.perf_counter() - t0) * 1000
        avg = total / 10
        print(f"\nN=16 평균 step: {avg:.1f}ms")
        assert avg < 500


# ── set_region + reset ───────────────────────────────────────────────

class TestGridUtils:
    def test_set_region(self):
        g = EnsembleGrid(N=8)
        g.set_region(0, 0, 4, 4, prior=0.9)
        m = g.state_matrix()
        assert m[:4, :4].mean() > 0.7

    def test_reset_clears_state(self):
        g = EnsembleGrid(N=8)
        for _ in range(5):
            g.step()
        g.reset(seed=42)
        assert g.frame == 0
        assert len(g._step_times) == 0

    def test_stats_keys(self):
        g = EnsembleGrid(N=8)
        g.step()
        st = g.stats()
        for key in ["frame", "mean_prior", "mean_coop",
                    "fractal_d", "last_step_ms"]:
            assert key in st

    def test_to_dict(self):
        g = EnsembleGrid(N=4)
        d = g.to_dict()
        assert d["N"] == 4
        assert len(d["agents"]) == 4


# ── 프랙탈 차원 ─────────────────────────────────────────────────────

class TestFractalDim:
    def test_uniform_matrix(self):
        # 균일 행렬 — 낮은 D 예상
        m = np.ones((32, 32)) * 0.8
        D = fractal_dim_boxcount(m, threshold=0.5)
        assert 1.0 <= D <= 2.0

    def test_random_matrix(self):
        # 랜덤 행렬 — 높은 D 예상
        m = np.random.rand(32, 32)
        D = fractal_dim_boxcount(m, threshold=0.5)
        assert 1.0 <= D <= 2.0

    def test_checkerboard(self):
        # 체커보드 — 복잡한 패턴
        m = np.zeros((32, 32))
        m[::2, ::2] = 1.0
        m[1::2, 1::2] = 1.0
        D = fractal_dim_boxcount(m, threshold=0.5)
        assert 1.0 <= D <= 2.0

    def test_multiscale_keys(self):
        m = np.random.rand(32, 32)
        result = fractal_dim_multiscale(m)
        assert "D_2scale" in result
        assert "D_3scale" in result
        assert "is_healthy" in result

    def test_history_alert(self):
        # D가 급락하면 alert=True
        history = [1.7] * 15 + [1.3]
        result = fractal_dim_history(history, window=10)
        assert result["alert"] is True

    def test_history_no_alert(self):
        history = [1.7] * 15 + [1.65]
        result = fractal_dim_history(history, window=10)
        assert result["alert"] is False
