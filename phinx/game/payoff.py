"""
phinx.game.payoff
-----------------
게임이론 보수 행렬 및 진화적 안정 전략(ESS) 판정.

지원 게임:
  - 죄수의 딜레마 (Prisoner's Dilemma)
  - Stag Hunt
  - Harmony Game
  - Snowdrift (Chicken)

ESS: 집단에서 충분히 많은 수가 채택하면
     외부 전략이 침투할 수 없는 전략.
     도시 시스템에서 '협력 규범의 자기강화 상태'.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ── 보수 행렬 정의 ───────────────────────────────────────────────────

@dataclass(frozen=True)
class PayoffMatrix:
    """
    2×2 대칭 게임 보수 행렬.

    보수 구조 (행위자 시점):
      R : 상호 협력 (Reward)
      T : 배반 성공 (Temptation)  — 상대가 협력할 때 나 배반
      S : 배반 당함 (Sucker)      — 내가 협력할 때 상대 배반
      P : 상호 배반 (Punishment)

    게임 유형 조건:
      PD       : T > R > P > S  (협력 딜레마)
      Stag Hunt: R > T > P > S  (조정 게임)
      Harmony  : R > T > S > P  (협력 우세)
      Snowdrift: T > R > S > P  (치킨 게임)
    """
    R: float  # Reward
    T: float  # Temptation
    S: float  # Sucker
    P: float  # Punishment
    name: str = "Custom"

    def matrix(self) -> np.ndarray:
        """2×2 보수 행렬 반환. 행=나, 열=상대. [협력, 배반]"""
        return np.array([
            [self.R, self.S],   # 내가 협력: 상대 협력→R, 상대 배반→S
            [self.T, self.P],   # 내가 배반: 상대 협력→T, 상대 배반→P
        ])

    def expected_payoff(self, my_coop: float,
                        other_coop: float) -> float:
        """
        혼합 전략 기댓값.

        Parameters
        ----------
        my_coop    : 나의 협력 확률 [0,1]
        other_coop : 상대 협력 확률 [0,1]
        """
        c, d = my_coop, 1.0 - my_coop
        oc, od = other_coop, 1.0 - other_coop
        return (c * oc * self.R + c * od * self.S
                + d * oc * self.T + d * od * self.P)

    def nash_equilibria(self) -> list[tuple[float, float]]:
        """
        순수 전략 내쉬균형 목록.
        (나의 협력확률, 상대의 협력확률) — 순수전략이므로 0 또는 1.
        """
        equilibria = []
        for s_me in [0.0, 1.0]:
            for s_other in [0.0, 1.0]:
                # 나의 최적 반응 확인
                payoff_coop = self.expected_payoff(1.0, s_other)
                payoff_defect = self.expected_payoff(0.0, s_other)
                me_ok = (s_me == 1.0 and payoff_coop >= payoff_defect) or \
                        (s_me == 0.0 and payoff_defect >= payoff_coop)
                # 상대의 최적 반응 확인
                payoff_coop2 = self.expected_payoff(1.0, s_me)
                payoff_defect2 = self.expected_payoff(0.0, s_me)
                other_ok = (s_other == 1.0 and payoff_coop2 >= payoff_defect2) or \
                           (s_other == 0.0 and payoff_defect2 >= payoff_coop2)
                if me_ok and other_ok:
                    equilibria.append((s_me, s_other))
        return equilibria

    def is_social_dilemma(self) -> bool:
        """사회적 딜레마 여부: 개인 최적 ≠ 집단 최적."""
        return self.T > self.R and self.P < self.R

    def to_dict(self) -> dict:
        return {"R": self.R, "T": self.T, "S": self.S,
                "P": self.P, "name": self.name}


# ── 사전 정의 게임 ───────────────────────────────────────────────────

PRISONERS_DILEMMA = PayoffMatrix(R=3.0, T=5.0, S=0.0, P=1.0,
                                  name="Prisoner's Dilemma")
STAG_HUNT         = PayoffMatrix(R=4.0, T=3.0, S=0.0, P=2.0,
                                  name="Stag Hunt")
HARMONY           = PayoffMatrix(R=4.0, T=3.0, S=2.0, P=1.0,
                                  name="Harmony")
SNOWDRIFT         = PayoffMatrix(R=3.0, T=4.0, S=1.0, P=0.0,
                                  name="Snowdrift")

GAMES = {
    "pd":        PRISONERS_DILEMMA,
    "stag_hunt": STAG_HUNT,
    "harmony":   HARMONY,
    "snowdrift": SNOWDRIFT,
}


# ── ESS 판정 ─────────────────────────────────────────────────────────

def is_ess(strategy: float, matrix: PayoffMatrix,
           epsilon: float = 1e-6) -> bool:
    """
    단일 전략이 ESS인지 판정.

    전략 s가 ESS: 모든 침입 전략 s' ≠ s에 대해
      f(s, s) > f(s', s)  또는
      f(s, s) = f(s', s) 이고 f(s, s') > f(s', s')

    Parameters
    ----------
    strategy : float  협력 확률 [0,1]
    matrix   : PayoffMatrix
    epsilon  : float  침입 전략 편차

    Returns
    -------
    bool
    """
    s = strategy
    invader = 1.0 - s  # 반대 순수전략으로 침입 테스트

    f_ss  = matrix.expected_payoff(s, s)
    f_is  = matrix.expected_payoff(invader, s)
    f_si  = matrix.expected_payoff(s, invader)
    f_ii  = matrix.expected_payoff(invader, invader)

    if f_ss > f_is + epsilon:
        return True
    if abs(f_ss - f_is) < epsilon:
        return f_si > f_ii + epsilon
    return False


def ess_landscape(matrix: PayoffMatrix,
                  resolution: int = 50) -> dict:
    """
    협력 확률 [0,1] 전체에서 ESS 지형 계산.

    Parameters
    ----------
    matrix     : PayoffMatrix
    resolution : int  샘플 수

    Returns
    -------
    dict:
        strategies  : ndarray  협력 확률값
        fitness     : ndarray  각 전략의 자기 대전 보수
        ess_points  : list     ESS 후보 전략값
    """
    strategies = np.linspace(0, 1, resolution)
    fitness = np.array([matrix.expected_payoff(s, s) for s in strategies])

    # ESS 후보: 순수 전략 (0, 1) + 혼합 전략 교점
    ess_pts = []
    for s in [0.0, 1.0]:
        if is_ess(s, matrix):
            ess_pts.append(s)

    # 혼합 전략 ESS: df/ds = 0 근방 탐색
    for i in range(len(strategies) - 1):
        if fitness[i] < fitness[i+1]:  # 단조 증가 구간 — 불안정
            continue
        ess_pts_candidate = strategies[i]
        if is_ess(float(ess_pts_candidate), matrix):
            ess_pts.append(float(ess_pts_candidate))

    return {
        "strategies": strategies,
        "fitness":    fitness,
        "ess_points": sorted(set(round(p, 3) for p in ess_pts)),
    }


def population_dynamics(matrix: PayoffMatrix,
                         initial_coop: float = 0.5,
                         steps: int = 100,
                         noise: float = 0.01) -> dict:
    """
    복제자 방정식 (Replicator Dynamics) 시뮬레이션.

    dc/dt = c(1-c)[f(c,c) - f(1-c,c)]

    집단의 협력율이 ESS로 수렴하는 과정을 시뮬레이션.
    도시 시스템에서 규범이 자기강화되는 과정의 수학적 모델.

    Parameters
    ----------
    matrix       : PayoffMatrix
    initial_coop : float  초기 협력율
    steps        : int    시뮬레이션 스텝 수
    noise        : float  로컬 ε 랜덤니스 (빗방울 불확정성)

    Returns
    -------
    dict:
        coop_history : ndarray  협력율 시계열
        converged_to : float    수렴값
        is_stable    : bool     안정 ESS 도달 여부
    """
    c = float(np.clip(initial_coop, 1e-6, 1 - 1e-6))
    history = [c]

    for _ in range(steps):
        f_coop   = matrix.expected_payoff(1.0, c)
        f_defect = matrix.expected_payoff(0.0, c)
        f_avg    = c * f_coop + (1 - c) * f_defect

        # 복제자 방정식 + 로컬 노이즈 (빗방울 ε)
        dc = c * (f_coop - f_avg) * 0.1
        epsilon = np.random.normal(0, noise)
        c = float(np.clip(c + dc + epsilon, 1e-6, 1 - 1e-6))
        history.append(c)

    converged = float(np.mean(history[-10:]))
    is_stable = float(np.std(history[-10:])) < 0.05

    return {
        "coop_history": np.array(history),
        "converged_to": converged,
        "is_stable":    is_stable,
    }


def pareto_efficiency(matrix: PayoffMatrix,
                       coop_rate: float) -> dict:
    """
    현재 협력율에서 파레토 효율성 평가.

    파레토 최적: 모두가 협력(R)일 때.
    현재 상태가 파레토 최적 대비 얼마나 비효율적인지 계산.
    """
    current = matrix.expected_payoff(coop_rate, coop_rate)
    optimal = matrix.R  # 모두 협력 시 보수
    worst   = matrix.P  # 모두 배반 시 보수

    denom = optimal - worst
    efficiency = (current - worst) / denom if denom > 0 else 0.0

    return {
        "current_payoff":  current,
        "pareto_optimal":  optimal,
        "efficiency":      float(np.clip(efficiency, 0, 1)),
        "is_pareto":       current >= optimal - 1e-6,
    }
