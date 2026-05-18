"""tests/test_output.py — Phase 3: output.realtime 테스트"""

import numpy as np
import pytest

from phinx.output.realtime import (
    OSCOutput, WebSocketOutput, RealtimeOutput, ConsoleOutput
)


# 샘플 Φ 결과
SAMPLE_RESULT = {
    "phi": 0.72,
    "S": 1.45,
    "D": 1.68,
    "T_star": 0.03,
    "F": -0.12,
    "Z": 8.4,
    "coop": 0.61,
    "is_critical": False,
    "signal": "stable",
    "compute_ms": 3.2,
    "frame": 42,
}


# ── OSCOutput ────────────────────────────────────────────────────────

class TestOSCOutput:
    def test_init_without_server(self):
        # 서버 없어도 초기화 가능
        osc = OSCOutput(host="127.0.0.1", port=19999)
        # python-osc 있으면 enabled, 없으면 disabled — 둘 다 OK
        assert isinstance(osc.is_enabled, bool)

    def test_send_count_starts_zero(self):
        osc = OSCOutput(port=19999)
        assert osc.sent_count == 0

    def test_send_returns_bool(self):
        osc = OSCOutput(port=19999)
        result = osc.send(SAMPLE_RESULT)
        assert isinstance(result, bool)


# ── RealtimeOutput ───────────────────────────────────────────────────

class TestRealtimeOutput:
    def test_init_no_channels(self):
        # OSC, WS 모두 None → 채널 없이 초기화
        rt = RealtimeOutput(osc_target=None, ws_port=None)
        assert rt.osc is None
        assert rt.ws is None

    def test_send_no_channels(self):
        rt = RealtimeOutput(osc_target=None, ws_port=None)
        result = rt.send(SAMPLE_RESULT)
        assert result["osc"] is False
        assert result["ws"] is False

    def test_send_count(self):
        rt = RealtimeOutput(osc_target=None, ws_port=None)
        rt.send(SAMPLE_RESULT)
        rt.send(SAMPLE_RESULT)
        assert rt.send_count == 2

    def test_disabled_skips_send(self):
        rt = RealtimeOutput(osc_target=None, ws_port=None)
        rt.enabled = False
        result = rt.send(SAMPLE_RESULT)
        assert result == {"osc": False, "ws": False}

    def test_on_send_callback(self):
        called = []
        rt = RealtimeOutput(
            osc_target=None, ws_port=None,
            on_send=lambda s: called.append(s)
        )
        rt.send(SAMPLE_RESULT)
        assert len(called) == 1
        assert "phi" in called[0]

    def test_stats_keys(self):
        rt = RealtimeOutput(osc_target=None, ws_port=None)
        rt.send(SAMPLE_RESULT)
        st = rt.stats()
        assert "send_count" in st
        assert "phi_mean" in st
        assert "phi_last" in st

    def test_stats_phi_correct(self):
        rt = RealtimeOutput(osc_target=None, ws_port=None)
        rt.send(SAMPLE_RESULT)
        assert abs(rt.stats()["phi_last"] - 0.72) < 1e-6

    def test_stop(self):
        rt = RealtimeOutput(osc_target=None, ws_port=None)
        rt.stop()
        assert not rt.enabled


# ── ConsoleOutput ────────────────────────────────────────────────────

class TestConsoleOutput:
    def test_send_returns_true(self, capsys):
        co = ConsoleOutput(every_n=1)
        result = co.send(SAMPLE_RESULT)
        assert result is True

    def test_every_n_filtering(self, capsys):
        co = ConsoleOutput(every_n=5)
        for _ in range(4):
            co.send(SAMPLE_RESULT)
        captured = capsys.readouterr()
        # 4번 중 every_n=5이므로 출력 없음
        assert captured.out == ""

    def test_every_n_prints(self, capsys):
        co = ConsoleOutput(every_n=3)
        for _ in range(3):
            co.send(SAMPLE_RESULT)
        captured = capsys.readouterr()
        assert "Φ=" in captured.out

    def test_critical_signal(self, capsys):
        co = ConsoleOutput(every_n=1)
        critical = dict(SAMPLE_RESULT)
        critical["signal"] = "critical"
        co.send(critical)
        captured = capsys.readouterr()
        assert "🔴" in captured.out

    def test_stable_signal(self, capsys):
        co = ConsoleOutput(every_n=1)
        co.send(SAMPLE_RESULT)
        captured = capsys.readouterr()
        assert "🟢" in captured.out


# ── 통합: PhiLoop + ConsoleOutput ───────────────────────────────────

class TestOutputIntegration:
    def test_philoop_with_console(self, capsys):
        import phinx
        np.random.seed(0)
        grid = phinx.EnsembleGrid(N=8)
        for i in range(grid.N):
            for j in range(grid.N):
                grid.agents[i][j].prior = float(np.random.rand())

        ensemble = phinx.ThermoEnsemble(grid, M=16)
        loop     = phinx.PhiLoop(grid, ensemble, fps=30)
        console  = ConsoleOutput(every_n=5)

        results = loop.run(n_frames=10, callback=console.send)
        assert len(results) == 10
        for r in results:
            assert 0 <= r["phi"] <= 1

    def test_realtime_output_in_loop(self):
        import phinx
        np.random.seed(0)
        grid = phinx.EnsembleGrid(N=8)
        for i in range(grid.N):
            for j in range(grid.N):
                grid.agents[i][j].prior = float(np.random.rand())

        ensemble = phinx.ThermoEnsemble(grid, M=16)
        rt       = RealtimeOutput(osc_target=None, ws_port=None)
        loop     = phinx.PhiLoop(grid, ensemble, fps=30)

        loop.run(n_frames=5, callback=rt.send)
        assert rt.send_count == 5

    def test_multiple_outputs_in_loop(self, capsys):
        import phinx
        np.random.seed(0)
        grid = phinx.EnsembleGrid(N=8)
        for i in range(grid.N):
            for j in range(grid.N):
                grid.agents[i][j].prior = float(np.random.rand())

        ensemble = phinx.ThermoEnsemble(grid, M=16)
        rt       = RealtimeOutput(osc_target=None, ws_port=None)
        console  = ConsoleOutput(every_n=3)

        def multi_output(result):
            rt.send(result)
            console.send(result)

        loop = phinx.PhiLoop(grid, ensemble, fps=30)
        loop.run(n_frames=6, callback=multi_output)

        assert rt.send_count == 6
