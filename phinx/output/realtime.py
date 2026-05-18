"""
phinx.output.realtime
---------------------
Φ 값을 외부 도구로 실시간 전송.

지원 프로토콜:
  OSC       : Max/MSP, TouchDesigner, SuperCollider, Ableton
  WebSocket : 브라우저 GLSL 쉐이더, p5.js, Three.js

OSC 메시지 스키마:
  /phinx/phi        f   [0.0, 1.0]   생존 지수
  /phinx/entropy    f   [0.0, ∞)     다양성
  /phinx/fractal    f   [1.0, 2.0]   패턴 복잡성
  /phinx/temp       f   [0.0, ∞)     유효온도
  /phinx/coop       f   [0.0, 1.0]   협력율
  /phinx/signal     s   str          "stable"|"warning"|"critical"
  /phinx/frame      i   int          프레임 번호
  /phinx/critical   i   [0, 1]       상전이 도달 여부
"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional


# ── OSC 선택적 의존성 ────────────────────────────────────────────────
try:
    from pythonosc import udp_client
    HAS_OSC = True
except ImportError:
    HAS_OSC = False


# ── WebSocket 선택적 의존성 ──────────────────────────────────────────
try:
    import asyncio
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False


class OSCOutput:
    """
    OSC UDP 출력.

    Parameters
    ----------
    host : str   대상 호스트. 기본값 "127.0.0.1"
    port : int   대상 포트. 기본값 9000.
                 Max/MSP: 9000, TouchDesigner: 9000, SC: 57120
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host = host
        self.port = port
        self._client = None
        self._enabled = False
        self._sent_count = 0

        if not HAS_OSC:
            print("[phinx.OSC] python-osc 미설치. "
                  "pip install python-osc 으로 설치하세요.")
            return

        try:
            self._client = udp_client.SimpleUDPClient(host, port)
            self._enabled = True
            print(f"[phinx.OSC] 연결 완료 → {host}:{port}")
        except Exception as e:
            print(f"[phinx.OSC] 연결 실패: {e}")

    def send(self, result: dict) -> bool:
        """
        Φ 결과를 OSC 메시지로 전송.

        Parameters
        ----------
        result : dict  compute_phi() 반환값

        Returns
        -------
        bool : 전송 성공 여부
        """
        if not self._enabled or self._client is None:
            return False

        try:
            self._client.send_message("/phinx/phi",
                                      float(result.get("phi", 0)))
            self._client.send_message("/phinx/entropy",
                                      float(result.get("S", 0)))
            self._client.send_message("/phinx/fractal",
                                      float(result.get("D", 1.5)))
            self._client.send_message("/phinx/temp",
                                      float(result.get("T_star", 0)))
            self._client.send_message("/phinx/coop",
                                      float(result.get("coop", 0.5)))
            self._client.send_message("/phinx/signal",
                                      str(result.get("signal", "stable")))
            self._client.send_message("/phinx/frame",
                                      int(result.get("frame", 0)))
            self._client.send_message("/phinx/critical",
                                      int(result.get("is_critical", False)))
            self._sent_count += 1
            return True
        except Exception as e:
            print(f"[phinx.OSC] 전송 오류: {e}")
            return False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def sent_count(self) -> int:
        return self._sent_count


class WebSocketOutput:
    """
    WebSocket JSON 출력.

    브라우저 p5.js / Three.js / GLSL 쉐이더와 연결.

    Parameters
    ----------
    host : str  바인딩 호스트. 기본값 "0.0.0.0"
    port : int  바인딩 포트. 기본값 8765.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: set = set()
        self._server = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._enabled = False
        self._latest: dict = {}
        self._sent_count = 0

        if not HAS_WS:
            print("[phinx.WS] websockets 미설치. "
                  "pip install websockets 으로 설치하세요.")
            return

        self._start_server()

    def _start_server(self) -> None:
        """별도 스레드에서 asyncio 루프 + WebSocket 서버 시작."""
        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        time.sleep(0.2)  # 서버 준비 대기
        self._enabled = True
        print(f"[phinx.WS] 서버 시작 → ws://{self.host}:{self.port}")

    async def _serve(self) -> None:
        async def handler(ws):
            self._clients.add(ws)
            try:
                # 연결 즉시 최신 상태 전송
                if self._latest:
                    await ws.send(json.dumps(self._latest))
                await ws.wait_closed()
            finally:
                self._clients.discard(ws)

        if HAS_WS:
            async with websockets.serve(handler, self.host, self.port):
                await asyncio.Future()  # 무한 대기

    def send(self, result: dict) -> bool:
        """
        Φ 결과를 연결된 모든 WebSocket 클라이언트에 전송.

        Parameters
        ----------
        result : dict  compute_phi() 반환값

        Returns
        -------
        bool : 전송 시도 여부
        """
        if not self._enabled or self._loop is None:
            return False

        payload = {
            "phi":      float(result.get("phi", 0)),
            "S":        float(result.get("S", 0)),
            "D":        float(result.get("D", 1.5)),
            "T_star":   float(result.get("T_star", 0)),
            "coop":     float(result.get("coop", 0.5)),
            "signal":   str(result.get("signal", "stable")),
            "frame":    int(result.get("frame", 0)),
            "critical": bool(result.get("is_critical", False)),
        }
        self._latest = payload

        if not self._clients:
            return True  # 연결된 클라이언트 없음 — 정상

        async def _broadcast():
            if self._clients:
                msg = json.dumps(payload)
                await asyncio.gather(
                    *[c.send(msg) for c in self._clients.copy()],
                    return_exceptions=True
                )

        asyncio.run_coroutine_threadsafe(_broadcast(), self._loop)
        self._sent_count += 1
        return True

    def stop(self) -> None:
        """서버 종료."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def sent_count(self) -> int:
        return self._sent_count


class RealtimeOutput:
    """
    OSC + WebSocket 통합 출력.

    하나의 send() 호출로 두 채널에 동시 전송.
    각 채널은 독립적으로 활성/비활성 가능.

    Parameters
    ----------
    osc_target  : tuple (host, port)  OSC 대상. None이면 비활성.
    ws_port     : int                 WebSocket 포트. None이면 비활성.
    on_send     : callable            매 전송 후 호출되는 콜백.
    """

    def __init__(
        self,
        osc_target: Optional[tuple] = ("127.0.0.1", 9000),
        ws_port: Optional[int] = None,
        on_send: Optional[Callable] = None,
    ):
        self.osc: Optional[OSCOutput] = None
        self.ws:  Optional[WebSocketOutput] = None
        self.on_send = on_send
        self._send_log: list[dict] = []
        self.enabled = True

        if osc_target is not None:
            self.osc = OSCOutput(host=osc_target[0], port=osc_target[1])

        if ws_port is not None:
            self.ws = WebSocketOutput(port=ws_port)

    def send(self, result: dict) -> dict:
        """
        Φ 결과를 활성화된 모든 채널에 전송.

        Parameters
        ----------
        result : dict  compute_phi() 반환값

        Returns
        -------
        dict : 전송 결과 요약
        """
        if not self.enabled:
            return {"osc": False, "ws": False}

        osc_ok = self.osc.send(result) if self.osc else False
        ws_ok  = self.ws.send(result)  if self.ws  else False

        summary = {
            "osc":    osc_ok,
            "ws":     ws_ok,
            "frame":  result.get("frame", 0),
            "phi":    result.get("phi", 0),
        }
        self._send_log.append(summary)

        if self.on_send:
            self.on_send(summary)

        return summary

    def stop(self) -> None:
        """모든 출력 채널 종료."""
        if self.ws:
            self.ws.stop()
        self.enabled = False

    @property
    def send_count(self) -> int:
        return len(self._send_log)

    def stats(self) -> dict:
        """전송 통계."""
        if not self._send_log:
            return {"send_count": 0}
        phis = [l["phi"] for l in self._send_log]
        return {
            "send_count": len(self._send_log),
            "phi_mean":   float(sum(phis) / len(phis)),
            "phi_last":   phis[-1] if phis else 0,
        }


class ConsoleOutput:
    """
    콘솔 출력 — 개발/디버깅용.
    OSC/WebSocket 없이도 실시간 상태 확인 가능.
    """

    def __init__(self, every_n: int = 10):
        """
        Parameters
        ----------
        every_n : int  N 프레임마다 출력. 기본값 10.
        """
        self.every_n = every_n
        self._count = 0

    def send(self, result: dict) -> bool:
        self._count += 1
        if self._count % self.every_n != 0:
            return True

        phi    = result.get("phi", 0)
        S      = result.get("S", 0)
        D      = result.get("D", 1.5)
        T      = result.get("T_star", 0)
        coop   = result.get("coop", 0.5)
        signal = result.get("signal", "stable")
        frame  = result.get("frame", 0)
        ms     = result.get("compute_ms", 0)

        bar_len = 20
        phi_bar = "█" * int(phi * bar_len) + "░" * (bar_len - int(phi * bar_len))

        signal_icon = {"stable": "🟢", "warning": "🟡", "critical": "🔴"}.get(
            signal, "⚪"
        )

        print(
            f"[{frame:05d}] {signal_icon} Φ={phi:.3f} [{phi_bar}] "
            f"S={S:.3f} D={D:.3f} T*={T:.4f} "
            f"coop={coop:.3f} {ms:.1f}ms"
        )
        return True
