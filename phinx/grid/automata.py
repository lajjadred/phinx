"""
phinx.grid.automata
-------------------
N×N 셀룰러 오토마타 그리드.

각 셀이 Agent를 보유하며, 이웃 반경 r 내에서
빗방울 충돌 + 베이즈 갱신을 프레임마다 실행합니다.

설계 원칙
- 이웃 반경 r=1 (8방향) 기본 → O(8N²) 연산/프레임
- 상태 행렬은 numpy 배열로 캐시 → 프랙탈·앙상블 계산에 재사용
- numba 있으면 핵심 루프 JIT 가속, 없으면 numpy fallback
"""

from __future__ import annotations

import time
import numpy as np
from typing import Optional

from phinx.core.agent import Agent, PAYOFF_PD
from phinx.grid.fractal import fractal_dim_boxcount


# ── numba 선택적 가속 ────────────────────────────────────────────────
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*a, **kw):
        def wrap(fn): return fn
        return wrap
    prange = range


# ── 이웃 오프셋 (8방향 + 중심 제외) ────────────────────────────────
_NEIGHBORS_R1 = np.array([
    [-1,-1],[-1, 0],[-1, 1],
    [ 0,-1],         [ 0, 1],
    [ 1,-1],[ 1, 0],[ 1, 1],
], dtype=np.int32)


class EnsembleGrid:
    """
    N×N 에이전트 그리드 — 셀룰러 오토마타 + 앙상블 기반.

    Parameters
    ----------
    N : int
        그리드 한 변의 크기. 총 N² 에이전트. 기본값 32.
    r : int
        이웃 반경. r=1 → 8방향, r=2 → 24방향. 기본값 1.
    payoff_matrix : dict
        게임이론 보수 행렬. 기본값 죄수의 딜레마.
    agent_kwargs : dict
        Agent 초기화 파라미터 기본값 override.
    wrap : bool
        경계 처리. True=토러스(주기 경계), False=고정 경계.
    """

    def __init__(
        self,
        N: int = 32,
        r: int = 1,
        payoff_matrix: dict = None,
        agent_kwargs: dict = None,
        wrap: bool = True,
    ):
        self.N = N
        self.r = r
        self.wrap = wrap
        self.payoff_matrix = payoff_matrix or dict(PAYOFF_PD)
        self._akw = agent_kwargs or {}

        # 에이전트 2D 배열
        self.agents: list[list[Agent]] = [
            [Agent(payoff_matrix=self.payoff_matrix, **self._akw)
             for _ in range(N)]
            for _ in range(N)
        ]

        # 성능 추적
        self._step_times: list[float] = []
        self.frame: int = 0

    # ── 핵심: 프레임 갱신 ───────────────────────────────────────────
    def step(self) -> float:
        """
        프레임 1회 전체 갱신.

        모든 셀에서 반경 r 이웃과 빗방울 충돌 처리.
        이웃 거리 = 유클리드 (대각선 ≈ 1.41).

        Returns
        -------
        float : 실행 시간 (ms)
        """
        t0 = time.perf_counter()
        N, r = self.N, self.r

        # 이웃 오프셋 생성 (r=1이면 캐시 사용)
        if r == 1:
            offsets = _NEIGHBORS_R1
        else:
            offsets = np.array(
                [[di, dj]
                 for di in range(-r, r+1)
                 for dj in range(-r, r+1)
                 if not (di == 0 and dj == 0)],
                dtype=np.int32
            )

        for i in range(N):
            for j in range(N):
                agent = self.agents[i][j]
                for di, dj in offsets:
                    ni = (i + di) % N if self.wrap else i + di
                    nj = (j + dj) % N if self.wrap else j + dj
                    if not self.wrap and (ni < 0 or ni >= N
                                          or nj < 0 or nj >= N):
                        continue
                    dist = (di*di + dj*dj) ** 0.5
                    neighbor = self.agents[ni][nj]
                    agent.meet(neighbor, distance=dist)

        elapsed = (time.perf_counter() - t0) * 1000  # ms
        self._step_times.append(elapsed)
        self.frame += 1
        return elapsed

    # ── 상태 행렬 추출 ──────────────────────────────────────────────
    def state_matrix(self) -> np.ndarray:
        """
        N×N prior 행렬 반환.
        각 셀의 Agent.prior 값. [0, 1]

        Returns
        -------
        ndarray shape (N, N)
        """
        return np.array(
            [[self.agents[i][j].prior for j in range(self.N)]
             for i in range(self.N)],
            dtype=np.float64
        )

    def cooperation_matrix(self) -> np.ndarray:
        """N×N 협력 성향 행렬."""
        return np.array(
            [[self.agents[i][j].cooperation_level() for j in range(self.N)]
             for i in range(self.N)],
            dtype=np.float64
        )

    def energy_matrix(self) -> np.ndarray:
        """N×N 내부 에너지 행렬 — 열역학 앙상블 입력."""
        return np.array(
            [[self.agents[i][j].energy for j in range(self.N)]
             for i in range(self.N)],
            dtype=np.float64
        )

    def epsilon_matrix(self) -> np.ndarray:
        """N×N 로컬 ε 분산 행렬 — 유효온도 T* 계산 입력."""
        return np.array(
            [[self.agents[i][j].epsilon_var for j in range(self.N)]
             for i in range(self.N)],
            dtype=np.float64
        )

    # ── 프랙탈 차원 ─────────────────────────────────────────────────
    def fractal_dim(self, threshold: float = 0.5,
                    scales: int = 3) -> float:
        """
        현재 상태 행렬의 프랙탈 차원 D.

        박스카운팅 근사 (2~3 스케일).
        D 정상범위 ≈ 1.6~1.8 (건강한 복잡계)
        D 급락 → 상전이 임계 신호.

        Parameters
        ----------
        threshold : float  이진화 임계값 (prior > threshold → 1)
        scales    : int    박스 스케일 수
        """
        mat = self.state_matrix()
        return fractal_dim_boxcount(mat, threshold=threshold, scales=scales)

    # ── 집계 통계 ───────────────────────────────────────────────────
    def stats(self) -> dict:
        """
        현재 프레임 집계 통계.

        Returns
        -------
        dict with keys:
            frame, mean_prior, std_prior, mean_coop,
            mean_energy, mean_epsilon, fractal_d,
            last_step_ms, avg_step_ms
        """
        s = self.state_matrix()
        c = self.cooperation_matrix()
        e = self.energy_matrix()
        eps = self.epsilon_matrix()

        return {
            "frame":        self.frame,
            "mean_prior":   float(s.mean()),
            "std_prior":    float(s.std()),
            "mean_coop":    float(c.mean()),
            "mean_energy":  float(e.mean()),
            "mean_epsilon": float(eps.mean()),
            "fractal_d":    self.fractal_dim(),
            "last_step_ms": self._step_times[-1] if self._step_times else 0.0,
            "avg_step_ms":  (float(np.mean(self._step_times))
                             if self._step_times else 0.0),
        }

    # ── 초기화 유틸 ─────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None) -> None:
        """그리드 전체 재초기화."""
        if seed is not None:
            np.random.seed(seed)
        self.agents = [
            [Agent(payoff_matrix=self.payoff_matrix, **self._akw)
             for _ in range(self.N)]
            for _ in range(self.N)
        ]
        self._step_times.clear()
        self.frame = 0

    def set_region(self, i0: int, j0: int, i1: int, j1: int,
                   prior: float) -> None:
        """
        특정 영역의 prior를 일괄 설정.
        초기 조건 커스터마이징용.
        """
        for i in range(i0, i1):
            for j in range(j0, j1):
                self.agents[i][j].prior = float(np.clip(prior, 1e-6, 1-1e-6))

    # ── 직렬화 ──────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """전체 그리드 상태 직렬화 — 저장/전송용."""
        return {
            "N": self.N,
            "r": self.r,
            "frame": self.frame,
            "agents": [
                [self.agents[i][j].to_dict() for j in range(self.N)]
                for i in range(self.N)
            ]
        }

    def __repr__(self) -> str:
        st = self.stats()
        return (f"EnsembleGrid(N={self.N}, r={self.r}, "
                f"frame={self.frame}, "
                f"mean_prior={st['mean_prior']:.3f}, "
                f"mean_coop={st['mean_coop']:.3f}, "
                f"D={st['fractal_d']:.3f})")
