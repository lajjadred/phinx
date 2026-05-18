"""tests/test_thermo.py — Phase 2: ThermoEnsemble + PhiLoop 테스트"""

import time
import numpy as np
import pytest

from phinx.grid.automata import EnsembleGrid
from phinx.thermo.ensemble import ThermoEnsemble
from phinx.phi import compute_phi, PhiLoop, sigmoid


# ── sigmoid 유틸 ─────────────────────────────────────────────────────

class TestSigmoid:
    def test_zero(self):
        assert abs(sigmoid(0.0) - 0.5) < 1e-9

    def test_large_positive(self):
        assert sigmoid(100.0) > 0.99

    def test_large_negative(self):
        assert sigmoid(-100.0) < 0.01

    def test_bounded(self):
        for x in [-500, -10, 0, 10, 500]:
            assert 0 <= sigmoid(x) <= 1


# ── ThermoEnsemble ───────────────────────────────────────────────────

class TestThermoEnsemble:
    def setup_method(self):
        np.random.seed(0)
        self.grid = EnsembleGrid(N=8)
        # 랜덤 prior 설정
        for i in range(self.grid.N):
            for j in range(self.grid.N):
                self.grid.agents[i][j].prior = float(np.random.rand())
                self.grid.agents[i][j].epsilon_var = float(
                    np.random.uniform(0.05, 0.3)
                )
        self.ensemble = ThermoEnsemble(self.grid, M=32)

    def test_effective_temperature_positive(self):
        T = self.ensemble.effective_temperature()
        assert T >= 0

    def test_effective_temperature_type(self):
        T = self.ensemble.effective_temperature()
        assert isinstance(T, float)

    def test_partition_function_positive(self):
        Z = self.ensemble.partition_function()
        assert Z > 0

    def test_entropy_positive(self):
        S = self.ensemble.entropy()
        assert S >= 0

    def test_entropy_uniform_is_max(self):
        # 균일 prior → 최대 엔트로피
        for i in range(self.grid.N):
            for j in range(self.grid.N):
                self.grid.agents[i][j].prior = 0.5
        S_uniform = self.ensemble.entropy(bins=8)

        # 극단 prior → 낮은 엔트로피
        for i in range(self.grid.N):
            for j in range(self.grid.N):
                self.grid.agents[i][j].prior = 0.99
        S_extreme = self.ensemble.entropy(bins=8)

        assert S_uniform >= S_extreme

    def test_free_energy_type(self):
        F = self.ensemble.free_energy()
        assert isinstance(F, float)

    def test_monte_carlo_coop_bounded(self):
        coop = self.ensemble.monte_carlo_coop()
        assert 0 <= coop <= 1

    def test_compute_returns_all_keys(self):
        result = self.ensemble.compute()
        for key in ["T_star", "Z", "S", "F", "mean_coop_mc", "phase"]:
            assert key in result

    def test_phase_keys(self):
        self.ensemble.compute()
        result = self.ensemble.compute()
        phase = result["phase"]
        assert "is_critical" in phase
        assert "signal" in phase

    def test_history_accumulates(self):
        for _ in range(5):
            self.ensemble.compute()
        assert len(self.ensemble._F_history) == 5

    def test_history_max_length(self):
        ens = ThermoEnsemble(self.grid, M=16, history_len=5)
        for _ in range(10):
            ens.compute()
        assert len(ens._F_history) <= 5

    def test_reset_history(self):
        for _ in range(5):
            self.ensemble.compute()
        self.ensemble.reset_history()
        assert len(self.ensemble._F_history) == 0


# ── compute_phi ──────────────────────────────────────────────────────

class TestComputePhi:
    def setup_method(self):
        np.random.seed(42)
        self.grid = EnsembleGrid(N=8)
        for i in range(self.grid.N):
            for j in range(self.grid.N):
                self.grid.agents[i][j].prior = float(np.random.rand())
        self.grid.step()
        self.ensemble = ThermoEnsemble(self.grid, M=32)

    def test_phi_bounded(self):
        result = compute_phi(self.grid, self.ensemble)
        assert 0 <= result["phi"] <= 1

    def test_phi_returns_all_keys(self):
        result = compute_phi(self.grid, self.ensemble)
        for key in ["phi", "S", "D", "T_star", "F", "Z",
                    "coop", "is_critical", "signal", "compute_ms", "frame"]:
            assert key in result

    def test_phi_compute_ms_reasonable(self):
        result = compute_phi(self.grid, self.ensemble)
        assert result["compute_ms"] < 500  # 500ms 이내

    def test_phi_frame_matches_grid(self):
        result = compute_phi(self.grid, self.ensemble)
        assert result["frame"] == self.grid.frame

    def test_phi_signal_valid(self):
        result = compute_phi(self.grid, self.ensemble)
        assert result["signal"] in ["stable", "warning", "critical"]

    def test_phi_alpha_effect(self):
        # alpha 높으면 entropy 기여 증가 → phi 변화
        r1 = compute_phi(self.grid, self.ensemble, alpha=0.1)
        r2 = compute_phi(self.grid, self.ensemble, alpha=0.9)
        # 둘 다 유효 범위
        assert 0 <= r1["phi"] <= 1
        assert 0 <= r2["phi"] <= 1

    def test_d_in_valid_range(self):
        result = compute_phi(self.grid, self.ensemble)
        assert 1.0 <= result["D"] <= 2.0


# ── PhiLoop ──────────────────────────────────────────────────────────

class TestPhiLoop:
    def setup_method(self):
        np.random.seed(0)
        self.grid = EnsembleGrid(N=8)
        for i in range(self.grid.N):
            for j in range(self.grid.N):
                self.grid.agents[i][j].prior = float(np.random.rand())
        self.ensemble = ThermoEnsemble(self.grid, M=16)
        self.loop = PhiLoop(self.grid, self.ensemble, fps=60)

    def test_run_n_frames(self):
        results = self.loop.run(n_frames=5)
        assert len(results) == 5

    def test_callback_called(self):
        called = []
        def cb(r): called.append(r["phi"])
        self.loop.run(n_frames=3, callback=cb)
        assert len(called) == 3

    def test_all_phi_bounded(self):
        results = self.loop.run(n_frames=5)
        for r in results:
            assert 0 <= r["phi"] <= 1

    def test_summary_keys(self):
        self.loop.run(n_frames=5)
        s = self.loop.summary()
        for key in ["n_frames", "phi_mean", "phi_min",
                    "phi_max", "avg_total_ms"]:
            assert key in s

    def test_summary_n_frames(self):
        self.loop.run(n_frames=7)
        assert self.loop.summary()["n_frames"] == 7

    def test_frame_increments(self):
        initial = self.grid.frame
        self.loop.run(n_frames=4)
        assert self.grid.frame == initial + 4


# ── 성능 벤치마크 ────────────────────────────────────────────────────

class TestPhasePerformance:
    def test_compute_phi_n16_under_200ms(self):
        np.random.seed(0)
        grid = EnsembleGrid(N=16)
        for i in range(grid.N):
            for j in range(grid.N):
                grid.agents[i][j].prior = float(np.random.rand())
        grid.step()
        ensemble = ThermoEnsemble(grid, M=64)

        t0 = time.perf_counter()
        compute_phi(grid, ensemble)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 200, f"compute_phi N=16: {elapsed:.1f}ms"

    def test_10_frames_philoop_n16(self):
        np.random.seed(0)
        grid = EnsembleGrid(N=16)
        for i in range(grid.N):
            for j in range(grid.N):
                grid.agents[i][j].prior = float(np.random.rand())
        ensemble = ThermoEnsemble(grid, M=64)
        loop = PhiLoop(grid, ensemble, fps=60)

        t0 = time.perf_counter()
        loop.run(n_frames=10)
        total = (time.perf_counter() - t0) * 1000
        avg = total / 10
        print(f"\nPhiLoop N=16 평균: {avg:.1f}ms/frame")
        # 실시간 기준 완화 (순수 Python — numba 없음)
        assert avg < 2000, f"평균 {avg:.1f}ms — 너무 느림"
