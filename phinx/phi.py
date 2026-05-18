"""
phinx.phi
---------
통합 생존함수 Φ — 전체 파이프라인의 출력.

Φ = sigmoid(α·S + β·D − γ·T*) · ⟨협력율⟩_M

Φ → 1 : 시스템 안정 (ESS 유지, 프랙탈 건강, 자유에너지 최소)
Φ → 0 : 시스템 붕괴 (Tc 도달, D 급락, 배반 ESS 전환)

α : 엔트로피(다양성) 가중치
β : 프랙탈 차원(복잡성) 가중치
γ : 유효온도(혼돈) 가중치

NEMAF 설치 작품에서 α·β·γ 가 미적 튜닝 파라미터.
"""

from __future__ import annotations

import time
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phinx.grid.automata import EnsembleGrid
    from phinx.thermo.ensemble import ThermoEnsemble


def sigmoid(x: float) -> float:
    """수치 안정 sigmoid."""
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


def compute_phi(
    grid: "EnsembleGrid",
    ensemble: "ThermoEnsemble",
    alpha: float = 0.3,
    beta: float = 0.4,
    gamma: float = 0.3,
    fractal_threshold: float = 0.5,
    fractal_scales: int = 3,
) -> dict:
    """
    통합 생존함수 Φ 계산.

    Parameters
    ----------
    grid      : EnsembleGrid
    ensemble  : ThermoEnsemble
    alpha     : float  엔트로피 가중치 (다양성)
    beta      : float  프랙탈 차원 가중치 (복잡성)
    gamma     : float  유효온도 가중치 (혼돈 — 음수 기여)
    fractal_threshold : float  프랙탈 이진화 임계
    fractal_scales    : int    박스카운팅 스케일 수

    Returns
    -------
    dict:
        phi          : float [0,1]  통합 생존 지수
        S            : float        엔트로피
        D            : float        프랙탈 차원
        T_star       : float        유효온도
        F            : float        자유에너지
        Z            : float        분배함수
        coop         : float        협력율 (몬테카를로)
        is_critical  : bool         상전이 임계 도달 여부
        signal       : str          "stable"|"warning"|"critical"
        compute_ms   : float        계산 시간 (ms)
        frame        : int          현재 프레임
    """
    t0 = time.perf_counter()

    # 열역학 앙상블 계산
    thermo = ensemble.compute()
    T_star = thermo["T_star"]
    S      = thermo["S"]
    F      = thermo["F"]
    Z      = thermo["Z"]
    coop   = thermo["mean_coop_mc"]
    phase  = thermo["phase"]

    # 프랙탈 차원
    D = grid.fractal_dim(
        threshold=fractal_threshold,
        scales=fractal_scales
    )

    # Φ 계산
    # sigmoid 내부: α·S + β·D − γ·T*
    # D를 [1,2] → [0,1] 정규화
    D_norm = (D - 1.0)
    score  = alpha * S + beta * D_norm - gamma * T_star
    phi    = sigmoid(score) * coop

    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "phi":         float(phi),
        "S":           S,
        "D":           D,
        "T_star":      T_star,
        "F":           F,
        "Z":           Z,
        "coop":        coop,
        "is_critical": phase["is_critical"],
        "signal":      phase["signal"],
        "compute_ms":  elapsed,
        "frame":       grid.frame,
    }


class PhiLoop:
    """
    실시간 루프 래퍼.

    grid.step() → compute_phi() → callback(result) 를
    지정 fps로 반복 실행.

    Parameters
    ----------
    grid     : EnsembleGrid
    ensemble : ThermoEnsemble
    fps      : int    목표 프레임레이트. 기본값 60.
    alpha, beta, gamma : float  Φ 튜닝 파라미터.
    """

    def __init__(
        self,
        grid: "EnsembleGrid",
        ensemble: "ThermoEnsemble",
        fps: int = 60,
        alpha: float = 0.3,
        beta: float = 0.4,
        gamma: float = 0.3,
    ):
        self.grid     = grid
        self.ensemble = ensemble
        self.fps      = fps
        self.alpha    = alpha
        self.beta     = beta
        self.gamma    = gamma
        self._running = False
        self.results: list[dict] = []

    def run(self, n_frames: int, callback=None) -> list[dict]:
        """
        n_frames 동안 루프 실행.

        Parameters
        ----------
        n_frames : int        실행 프레임 수
        callback : callable   매 프레임 결과를 받는 함수
                              callback(result: dict) -> None

        Returns
        -------
        list[dict] : 프레임별 결과 목록
        """
        frame_budget = 1.0 / self.fps * 1000  # ms
        self.results = []
        self._running = True

        for _ in range(n_frames):
            if not self._running:
                break

            t0 = time.perf_counter()

            # 1. 그리드 갱신
            self.grid.step()

            # 2. Φ 계산
            result = compute_phi(
                self.grid, self.ensemble,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

            # 3. 콜백
            if callback is not None:
                callback(result)

            self.results.append(result)

            # 4. 프레임 예산 체크
            elapsed = (time.perf_counter() - t0) * 1000
            result["total_ms"] = elapsed
            result["budget_ok"] = elapsed <= frame_budget

        self._running = False
        return self.results

    def stop(self) -> None:
        """루프 중단."""
        self._running = False

    def summary(self) -> dict:
        """실행 결과 요약 통계."""
        if not self.results:
            return {}
        phis   = [r["phi"]    for r in self.results]
        totals = [r.get("total_ms", 0) for r in self.results]
        budgets = [r.get("budget_ok", True) for r in self.results]

        return {
            "n_frames":       len(self.results),
            "phi_mean":       float(np.mean(phis)),
            "phi_min":        float(np.min(phis)),
            "phi_max":        float(np.max(phis)),
            "phi_final":      phis[-1],
            "avg_total_ms":   float(np.mean(totals)),
            "max_total_ms":   float(np.max(totals)),
            "budget_hit_rate": float(np.mean(budgets)),
            "signals":        [r["signal"] for r in self.results],
        }
