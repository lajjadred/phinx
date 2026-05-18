"""
phinx.core.agent
----------------
개체 상태 벡터 ψᵢ = (s, π, P, ε, E)

s  : 행동 성향 float [0,1]          ← 행동심리 / 성격이론
π  : 전략 분포 ndarray [coop, defect] ← 게임이론
P  : 베이즈 사전확률 float [0,1]     ← 베이즈주의
ε  : 로컬 노이즈 분산 float          ← 빗방울 랜덤니스 (글로벌 T 대체)
E  : 내부 에너지 float               ← 심신 정동 수준 (열역학)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── 게임이론 보수 행렬 상수 ──────────────────────────────────────────
PAYOFF_PD = {"R": 3.0, "T": 5.0, "S": 0.0, "P": 1.0}   # 죄수의 딜레마
PAYOFF_SH = {"R": 4.0, "T": 3.0, "S": 0.0, "P": 2.0}   # Stag Hunt
PAYOFF_HG = {"R": 4.0, "T": 3.0, "S": 2.0, "P": 1.0}   # Harmony


@dataclass
class Agent:
    """
    단일 개체 — 모든 이론의 교차점.

    Parameters
    ----------
    prior : float
        베이즈 사전확률 P(H). 협력 가능성에 대한 초기 믿음. [0, 1]
    strategy : ndarray, optional
        [협력 성향, 배반 성향]. 합이 1. 기본값 [0.5, 0.5].
    epsilon_var : float
        로컬 노이즈 ε 의 분산. 글로벌 temperature를 대체.
        충돌 시 이웃과 공진화.
    energy : float, optional
        내부 에너지 Eᵢ. 심신 정동 수준. 기본값 random [0,1].
    payoff_matrix : dict, optional
        게임이론 보수 행렬. 기본값 죄수의 딜레마.
    tau : int
        베이즈 갱신 반복 횟수. 기본값 3. (수렴-연산량 균형)
    """

    prior: float = 0.5
    strategy: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    epsilon_var: float = 0.1
    energy: float = field(default_factory=lambda: float(np.random.rand()))
    payoff_matrix: dict = field(default_factory=lambda: dict(PAYOFF_PD))
    tau: int = 3

    # 내부 상태 (직렬화 제외)
    _history: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.prior = float(np.clip(self.prior, 1e-6, 1 - 1e-6))
        self.strategy = np.array(self.strategy, dtype=float)
        self.strategy /= self.strategy.sum()  # 정규화

    # ── 핵심: 빗방울 충돌 ───────────────────────────────────────────
    def meet(self, other: "Agent", distance: float) -> bool:
        """
        빗방울 충돌 이벤트.

        P(충돌) = exp(−d / r₀) — 거리 기반 지수 감쇠.
        충돌 시:
          1. 베이즈 갱신 (τ회)
          2. 전략 갱신 (게임이론 보수)
          3. ε 분산 공진화

        Returns
        -------
        bool : 실제 충돌이 일어났는지 여부
        """
        p_collision = np.exp(-distance)
        if np.random.rand() > p_collision:
            return False

        # 1. 베이즈 갱신 ── τ회 반복
        for _ in range(self.tau):
            likelihood = other.prior          # 상대 상태가 증거 E
            denom = (likelihood * self.prior
                     + (1.0 - likelihood) * (1.0 - self.prior))
            if denom > 1e-9:
                self.prior = (likelihood * self.prior) / denom
            self.prior = float(np.clip(self.prior, 1e-6, 1 - 1e-6))

        # 2. 전략 갱신 ── 게임이론 보수
        delta = self._payoff_delta(other)
        self.strategy = np.clip(
            self.strategy * 0.9 + delta * 0.1, 1e-6, None
        )
        self.strategy /= self.strategy.sum()

        # 3. ε 분산 공진화
        self.epsilon_var = (0.8 * self.epsilon_var
                            + 0.2 * other.epsilon_var)

        # 4. 에너지 교환 (열역학)
        self.energy = 0.95 * self.energy + 0.05 * other.energy

        self._history.append(self.prior)
        return True

    def act(self) -> float:
        """
        로컬 ε 샘플링 포함 행동 출력.

        글로벌 temperature 없이 개체별 분산으로 불확정성 구현.
        εᵢ ~ N(0, ε_var) — 빗방울 수열의 확률적 불확정성.

        Returns
        -------
        float : 행동값 [0, 1]
        """
        epsilon = np.random.normal(0.0, np.sqrt(max(self.epsilon_var, 1e-9)))
        return float(np.clip(self.prior + epsilon, 0.0, 1.0))

    def cooperation_level(self) -> float:
        """협력 성향 — 전략 벡터의 협력 성분."""
        return float(self.strategy[0])

    # ── 내부 유틸 ────────────────────────────────────────────────────
    def _payoff_delta(self, other: "Agent") -> np.ndarray:
        """게임이론 보수를 전략 업데이트 델타로 변환."""
        m = self.payoff_matrix
        c_me = self.strategy[0]
        c_ot = other.strategy[0]

        payoff = (c_me * c_ot * m["R"]
                  + c_me * (1 - c_ot) * m["S"]
                  + (1 - c_me) * c_ot * m["T"]
                  + (1 - c_me) * (1 - c_ot) * m["P"])

        max_payoff = max(v for k, v in m.items() if k != "name")
        norm = payoff / max_payoff if max_payoff > 0 else 0.5
        return np.array([norm, 1.0 - norm])

    # ── 직렬화 ───────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """상태 직렬화 — OSC/WebSocket 전송, 저장용."""
        return {
            "prior": self.prior,
            "strategy": self.strategy.tolist(),
            "epsilon_var": self.epsilon_var,
            "energy": self.energy,
            "cooperation": self.cooperation_level(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Agent":
        return cls(
            prior=d["prior"],
            strategy=np.array(d["strategy"]),
            epsilon_var=d["epsilon_var"],
            energy=d["energy"],
        )

    # ── 수렴 진단 ────────────────────────────────────────────────────
    def kl_divergence_from_history(self) -> Optional[float]:
        """
        최근 갱신의 KL 다이버전스.
        낮을수록 prior가 수렴 중. δ < 0.01 이면 수렴 판단.
        """
        if len(self._history) < 2:
            return None
        p = self._history[-1]
        q = self._history[-2]
        p = np.clip(p, 1e-9, 1 - 1e-9)
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return float(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))
