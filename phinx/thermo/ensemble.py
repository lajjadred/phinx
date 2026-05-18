"""
phinx.thermo.ensemble
---------------------
열역학 앙상블 레이어 — 미시↔거시 인터페이스.

핵심 개념:
  Z   : 분배함수 — 전체 상태 확률 합산
  T*  : 유효온도 — 로컬 ε 분산에서 창발 (글로벌 T 대체)
  S   : 엔트로피 — 다양성 척도
  F   : 자유에너지 — 안정성 척도 (F 최소 = ESS)
  Tc  : 상전이 임계 — 붕괴 예측 신호

미시(Agent.ε) → 앙상블(Z, T*, S, F) → 거시(Φ, D, ⟨협력율⟩)
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phinx.grid.automata import EnsembleGrid


class ThermoEnsemble:
    """
    EnsembleGrid 위에 올라가는 열역학 앙상블 계층.

    Parameters
    ----------
    grid : EnsembleGrid
        대상 그리드.
    M : int
        몬테카를로 샘플 수. 기본값 64.
        오차 ~ 1/√M. M=64 → ~12% 오차, ~2ms 처리.
    k : float
        볼츠만 상수 역할 스케일 파라미터. 기본값 1.0.
    history_len : int
        상전이 감지용 히스토리 길이. 기본값 30.
    """

    def __init__(
        self,
        grid: "EnsembleGrid",
        M: int = 64,
        k: float = 1.0,
        history_len: int = 30,
    ):
        self.grid = grid
        self.M = M
        self.k = k
        self.history_len = history_len

        # 히스토리 (상전이 감지용)
        self._F_history: list[float] = []
        self._S_history: list[float] = []
        self._T_history: list[float] = []

    # ── 유효온도 T* ──────────────────────────────────────────────────
    def effective_temperature(self) -> float:
        """
        T* = Var(εᵢ) / k

        로컬 ε 분산의 집합적 효과 → 창발적 온도.
        글로벌 temperature 파라미터 불필요.

        T* 높음 → 시스템 혼돈 (불확정성 큼)
        T* 낮음 → 시스템 질서 (행동 수렴)
        """
        eps = self.grid.epsilon_matrix().flatten()
        return float(np.var(eps) / self.k)

    # ── 분배함수 Z ───────────────────────────────────────────────────
    def partition_function(self, T_star: float = None) -> float:
        """
        Z = Σᵢ exp(−Eᵢ / kT*)

        T* 가 0에 가까우면 Z → 최저 에너지 상태만 선택.
        T* 가 크면 Z → 모든 상태 동등.

        Parameters
        ----------
        T_star : float, optional  유효온도. None이면 자동 계산.
        """
        if T_star is None:
            T_star = self.effective_temperature()

        energies = self.grid.energy_matrix().flatten()
        kT = self.k * max(T_star, 1e-9)
        weights = np.exp(-energies / kT)
        return float(weights.sum())

    # ── 엔트로피 S ───────────────────────────────────────────────────
    def entropy(self, bins: int = 8) -> float:
        """
        S = −k Σ pᵢ ln pᵢ

        8빈 히스토그램 근사 → O(N) 경량 계산.
        실시간 처리 전제.

        S 높음 → prior 분포 다양 (성격 다양성, 탄력성)
        S 낮음 → prior 분포 집중 (동질화, 취약)
        """
        priors = self.grid.state_matrix().flatten()
        hist, _ = np.histogram(priors, bins=bins, range=(0, 1))
        probs = hist / hist.sum()
        probs = probs[probs > 0]  # 0 제거 (log 안전)
        return float(-self.k * np.sum(probs * np.log(probs)))

    # ── 자유에너지 F ─────────────────────────────────────────────────
    def free_energy(self, T_star: float = None,
                    S: float = None) -> float:
        """
        F = ⟨E⟩ − T*·S

        F 최소 → 안정 상태 (ESS와 동형).
        F 증가 → 불안정, 상전이 임박.

        열역학 제2법칙: 자발적 과정은 F를 낮추는 방향.
        도시 시스템: 협력 규범 정착 = F 최솟값 탐색.
        """
        if T_star is None:
            T_star = self.effective_temperature()
        if S is None:
            S = self.entropy()

        mean_E = float(self.grid.energy_matrix().mean())
        return mean_E - T_star * S

    # ── 몬테카를로 앙상블 추정 ───────────────────────────────────────
    def monte_carlo_coop(self) -> float:
        """
        M개 몬테카를로 샘플로 협력율 추정.
        ⟨협력율⟩_M = (1/M) Σ act()

        각 에이전트의 로컬 ε 포함 샘플링.
        오차 ~ 1/√M.
        """
        N = self.grid.N
        agents_flat = [self.grid.agents[i][j]
                       for i in range(N) for j in range(N)]

        # M개 샘플 (에이전트 수 초과 시 복원 추출)
        n_agents = len(agents_flat)
        indices = np.random.choice(n_agents, size=self.M, replace=True)
        samples = [agents_flat[idx].act() for idx in indices]
        return float(np.mean(samples))

    # ── 상전이 감지 ──────────────────────────────────────────────────
    def detect_phase_transition(self, window: int = 10) -> dict:
        """
        슬라이딩 윈도우로 상전이 임계 Tc 감지.

        ∂²F/∂T*² ≈ 0 조건 → 히스토리 곡률로 근사.

        Returns
        -------
        dict:
            is_critical : bool   상전이 임계 도달 여부
            F_trend     : float  자유에너지 변화율
            S_trend     : float  엔트로피 변화율
            signal      : str    "stable" | "warning" | "critical"
        """
        if len(self._F_history) < window + 1:
            return {"is_critical": False, "F_trend": 0.0,
                    "S_trend": 0.0, "signal": "stable"}

        F_recent = np.array(self._F_history[-window:])
        S_recent = np.array(self._S_history[-window:])

        # 선형 추세 기울기
        x = np.arange(window)
        F_trend = float(np.polyfit(x, F_recent, 1)[0])
        S_trend = float(np.polyfit(x, S_recent, 1)[0])

        # 2차 미분 근사 (곡률)
        if len(self._F_history) >= window * 2:
            F_old = np.array(self._F_history[-window*2:-window])
            F_cur = F_recent
            curvature = float(np.mean(F_cur) - np.mean(F_old))
        else:
            curvature = 0.0

        # 판정 기준
        is_critical = (F_trend > 0.01 and S_trend < -0.01)
        signal = "stable"
        if abs(F_trend) > 0.005 or abs(S_trend) > 0.005:
            signal = "warning"
        if is_critical or abs(curvature) > 0.05:
            signal = "critical"

        return {
            "is_critical": is_critical,
            "F_trend":     F_trend,
            "S_trend":     S_trend,
            "curvature":   curvature,
            "signal":      signal,
        }

    # ── 전체 앙상블 계산 (프레임당 호출) ────────────────────────────
    def compute(self) -> dict:
        """
        프레임 1회 전체 열역학 양 계산.

        히스토리 자동 업데이트.

        Returns
        -------
        dict:
            T_star      : 유효온도
            Z           : 분배함수
            S           : 엔트로피
            F           : 자유에너지
            mean_coop_mc: 몬테카를로 협력율 추정
            phase       : 상전이 감지 결과
        """
        T_star = self.effective_temperature()
        Z      = self.partition_function(T_star)
        S      = self.entropy()
        F      = self.free_energy(T_star, S)
        coop   = self.monte_carlo_coop()
        
        # 히스토리 갱신
        self._F_history.append(F)
        self._S_history.append(S)
        self._T_history.append(T_star)

        # 히스토리 길이 제한
        if len(self._F_history) > self.history_len:
            self._F_history.pop(0)
            self._S_history.pop(0)
            self._T_history.pop(0)

        phase = self.detect_phase_transition()

        return {
            "T_star":       T_star,
            "Z":            Z,
            "S":            S,
            "F":            F,
            "mean_coop_mc": coop,
            "phase":        phase,
        }

    def reset_history(self) -> None:
        """히스토리 초기화."""
        self._F_history.clear()
        self._S_history.clear()
        self._T_history.clear()
