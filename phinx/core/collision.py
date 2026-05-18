"""
phinx.core.collision
--------------------
빗방울 수열 충돌 확률 계산 및 배치 처리.

P(적어도 하나의 충돌) = 1 − (1−p)^C(n,2)
단일 쌍: p = exp(−d / r₀)

이 모듈은 에이전트 쌍의 충돌을 배치로 처리하며,
numba가 설치된 경우 JIT 가속을 자동 적용합니다.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phinx.core.agent import Agent

# numba 선택적 가속
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):           # fallback decorator
        def wrapper(fn): return fn
        return wrapper


# ── 핵심 확률 함수 ──────────────────────────────────────────────────

def collision_prob(distance: float, r0: float = 1.0) -> float:
    """
    단일 쌍 충돌 확률.

    p = exp(−d / r₀)

    Parameters
    ----------
    distance : float  유클리드 거리
    r0       : float  특성 반경 (기본값 1.0 = 이웃 1칸)
    """
    return float(np.exp(-distance / r0))


def ensemble_collision_prob(p: float, n: int) -> float:
    """
    n개 개체 중 적어도 하나의 충돌 확률.

    P = 1 − (1−p)^C(n,2)

    Parameters
    ----------
    p : float  단일 쌍 충돌 확률
    n : int    개체 수
    """
    pairs = n * (n - 1) // 2
    return float(1.0 - (1.0 - p) ** pairs)


# ── 이웃 거리 행렬 계산 (numba 가속) ────────────────────────────────

@njit(cache=True)
def _pairwise_distances_numba(positions: np.ndarray) -> np.ndarray:
    """n×n 거리 행렬 (numba JIT)."""
    n = positions.shape[0]
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            d = (dx * dx + dy * dy) ** 0.5
            dists[i, j] = d
            dists[j, i] = d
    return dists


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """
    에이전트 위치 배열에서 거리 행렬 계산.

    Parameters
    ----------
    positions : ndarray shape (n, 2)  각 에이전트의 (x, y) 위치

    Returns
    -------
    ndarray shape (n, n)  유클리드 거리 행렬
    """
    if HAS_NUMBA:
        return _pairwise_distances_numba(positions.astype(np.float64))
    # fallback: scipy/numpy
    diff = positions[:, None, :] - positions[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


# ── 배치 충돌 처리 ──────────────────────────────────────────────────

def process_collisions(
    agents: list["Agent"],
    positions: np.ndarray,
    r: int = 1,
    r0: float = 1.0,
) -> int:
    """
    이웃 반경 r 내 에이전트 쌍에 대해 충돌 처리.

    Grid 외부에서 임의 위치 에이전트 목록을 받을 때 사용.
    EnsembleGrid.step()은 내부에서 직접 처리하므로 선택적 사용.

    Parameters
    ----------
    agents    : list[Agent]
    positions : ndarray (n, 2)  에이전트 위치
    r         : int             이웃 반경 (셀 단위)
    r0        : float           충돌 확률 특성 반경

    Returns
    -------
    int : 실제 충돌 횟수
    """
    n = len(agents)
    if n < 2:
        return 0

    dists = pairwise_distances(positions)
    collision_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d <= r:
                hit = agents[i].meet(agents[j], distance=d / r0)
                if hit:
                    # 대칭 갱신 (j→i 방향도 처리)
                    agents[j].meet(agents[i], distance=d / r0)
                    collision_count += 1

    return collision_count


# ── 진단 유틸 ────────────────────────────────────────────────────────

def collision_stats(agents: list["Agent"], positions: np.ndarray,
                    r: int = 1) -> dict:
    """
    충돌 통계 반환 — 디버깅 및 앙상블 진단용.

    Returns
    -------
    dict with keys:
        n_agents      : 전체 에이전트 수
        n_pairs       : 이웃 쌍 수
        expected_hits : 기댓값 충돌 횟수
        mean_distance : 평균 이웃 거리
    """
    dists = pairwise_distances(positions)
    mask = (dists > 0) & (dists <= r)
    n_pairs = mask.sum() // 2
    near_dists = dists[mask]

    if len(near_dists) == 0:
        return {"n_agents": len(agents), "n_pairs": 0,
                "expected_hits": 0.0, "mean_distance": 0.0}

    mean_d = float(near_dists.mean())
    p_avg = collision_prob(mean_d)
    expected = p_avg * n_pairs

    return {
        "n_agents": len(agents),
        "n_pairs": int(n_pairs),
        "expected_hits": float(expected),
        "mean_distance": mean_d,
    }
