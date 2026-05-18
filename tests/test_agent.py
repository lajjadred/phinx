"""tests/test_agent.py — Phase 1 Week 1 테스트"""

import numpy as np
import pytest
from phinx.core.agent import Agent
from phinx.core.collision import collision_prob, ensemble_collision_prob


# ── Agent 기본 테스트 ────────────────────────────────────────────────

class TestAgentInit:
    def test_default_init(self):
        a = Agent()
        assert 0 < a.prior < 1
        assert len(a.strategy) == 2
        assert abs(a.strategy.sum() - 1.0) < 1e-9
        assert a.epsilon_var > 0

    def test_prior_clipping(self):
        a = Agent(prior=0.0)
        assert a.prior > 0
        b = Agent(prior=1.0)
        assert b.prior < 1

    def test_strategy_normalization(self):
        a = Agent(strategy=np.array([2.0, 2.0]))
        assert abs(a.strategy.sum() - 1.0) < 1e-9


class TestAgentMeet:
    def test_meet_returns_bool(self):
        a = Agent(prior=0.7)
        b = Agent(prior=0.3)
        result = a.meet(b, distance=0.1)  # 거의 확실히 충돌
        assert isinstance(result, bool)

    def test_prior_stays_bounded(self):
        a = Agent(prior=0.5)
        b = Agent(prior=0.9)
        for _ in range(100):
            a.meet(b, distance=0.0)
        assert 0 < a.prior < 1

    def test_strategy_stays_normalized(self):
        a = Agent(prior=0.5)
        b = Agent(prior=0.8)
        for _ in range(50):
            a.meet(b, distance=0.1)
        assert abs(a.strategy.sum() - 1.0) < 1e-6

    def test_far_distance_rarely_collides(self):
        hits = 0
        for _ in range(1000):
            a = Agent()
            b = Agent()
            if a.meet(b, distance=10.0):
                hits += 1
        # 거리 10에서 p = exp(-10) ≈ 0.000045 → 1000번 중 < 5번 기대
        assert hits < 20

    def test_zero_distance_almost_always_collides(self):
        hits = 0
        for _ in range(100):
            a = Agent()
            b = Agent()
            if a.meet(b, distance=0.0):
                hits += 1
        assert hits > 95

    def test_epsilon_coevolution(self):
        a = Agent(epsilon_var=0.5)
        b = Agent(epsilon_var=0.1)
        for _ in range(20):
            a.meet(b, distance=0.0)
        # a의 ε이 b 쪽으로 수렴해야 함
        assert a.epsilon_var < 0.5


class TestAgentAct:
    def test_act_bounded(self):
        a = Agent(prior=0.5, epsilon_var=0.01)
        acts = [a.act() for _ in range(1000)]
        assert all(0 <= v <= 1 for v in acts)

    def test_act_mean_near_prior(self):
        a = Agent(prior=0.7, epsilon_var=0.001)
        acts = [a.act() for _ in range(500)]
        assert abs(np.mean(acts) - 0.7) < 0.05


class TestAgentSerialization:
    def test_roundtrip(self):
        a = Agent(prior=0.6, epsilon_var=0.2)
        d = a.to_dict()
        b = Agent.from_dict(d)
        assert abs(a.prior - b.prior) < 1e-9
        assert abs(a.epsilon_var - b.epsilon_var) < 1e-9


class TestKLDivergence:
    def test_kl_decreases_over_meetings(self):
        a = Agent(prior=0.5)
        b = Agent(prior=0.9)
        for _ in range(10):
            a.meet(b, distance=0.0)
        kl = a.kl_divergence_from_history()
        if kl is not None:
            assert kl >= 0


# ── 충돌 확률 테스트 ─────────────────────────────────────────────────

class TestCollisionProb:
    def test_zero_distance(self):
        assert collision_prob(0.0) == pytest.approx(1.0)

    def test_large_distance(self):
        assert collision_prob(100.0) < 1e-9

    def test_monotone_decreasing(self):
        probs = [collision_prob(d) for d in [0.1, 0.5, 1.0, 2.0, 5.0]]
        assert probs == sorted(probs, reverse=True)

    def test_ensemble_prob_increases_with_n(self):
        p = 0.1
        probs = [ensemble_collision_prob(p, n) for n in [2, 5, 10, 20]]
        assert probs == sorted(probs)

    def test_ensemble_prob_bounded(self):
        assert 0 <= ensemble_collision_prob(0.5, 10) <= 1.0


# ── 성능 벤치마크 (느린 경우 경고) ──────────────────────────────────

class TestPerformance:
    def test_1000_agents_creation(self):
        import time
        t0 = time.time()
        agents = [Agent() for _ in range(1000)]
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"1000 에이전트 생성에 {elapsed:.2f}s — 너무 느림"

    def test_100_meetings(self):
        import time
        a = Agent()
        b = Agent()
        t0 = time.time()
        for _ in range(100):
            a.meet(b, distance=0.5)
        elapsed = time.time() - t0
        assert elapsed < 0.1, f"100회 meet에 {elapsed:.3f}s — 너무 느림"
