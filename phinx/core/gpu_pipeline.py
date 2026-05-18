"""
phinx/core/gpu_pipeline.py

전체 파이프라인 GPU화
  Grid Step     → Triton 커널 (에이전트 상호작용)
  Ensemble      → torch GPU 병렬 (M개 동시 샘플)
  Φ 계산        → GPU 텐서 연산
  Output        → CPU로 비동기 전송

CC 7.5 (RTX 20xx) ~ CC 8.6 (RTX 30xx) 동시 지원
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from .attention_kernel import agent_attention, masked_agent_attention, _get_cc
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from attention_kernel import agent_attention, masked_agent_attention, _get_cc


# ─────────────────────────────────────────────
# 설정 dataclass
# ─────────────────────────────────────────────

@dataclass
class PhinxConfig:
    """
    전체 파이프라인 설정
    GPU 모델에 따라 자동 최적화
    """
    # 그리드
    N: int = 32              # 그리드 크기 N×N
    r: float = 1.5           # 이웃 반경
    D: int = 16              # 에이전트 상태 차원

    # 열역학 앙상블
    M: int = 64              # 앙상블 샘플 수
    k_boltzmann: float = 1.0

    # 생존함수 Φ 튜닝
    alpha: float = 0.3       # 엔트로피(다양성) 가중치
    beta: float  = 0.4       # 프랙탈(복잡성) 가중치
    gamma: float = 0.3       # 유효온도(혼돈) 가중치

    # 게임이론
    payoff: dict = field(default_factory=lambda: {
        'R': 3.0, 'T': 5.0, 'S': 0.0, 'P': 1.0
    })

    # 출력
    fps_target: int = 60

    def auto_tune(self):
        """GPU 모델에 따라 파라미터 자동 최적화"""
        cc = _get_cc()
        if not torch.cuda.is_available():
            self.M = 32      # CPU fallback — 경량화
            return

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        # CC 7.5 (RTX 20xx, 8GB) 기준
        if cc == (7, 5):
            self.M   = min(self.M, 128)
            self.N   = min(self.N, 32)
            self.D   = min(self.D, 32)

        # CC 8.6 (RTX 3060, 12GB)
        elif cc[0] == 8:
            if vram_gb >= 10:
                self.M = min(self.M * 2, 512)
                self.N = min(self.N * 2, 64)

        print(f"[phinx] GPU: CC {cc[0]}.{cc[1]}, VRAM {vram_gb:.1f}GB")
        print(f"[phinx] 자동 설정 → N={self.N}, M={self.M}, D={self.D}")


# ─────────────────────────────────────────────
# GPU 에이전트 상태 (텐서 기반)
# ─────────────────────────────────────────────

class GPUAgentState:
    """
    N×N 에이전트 전체를 GPU 텐서로 관리
    개별 Python 객체 루프 제거 → 배치 연산

    상태 텐서:
      prior      [N², 1]   베이즈 사전확률 P(H)
      strategy   [N², 2]   협력/배반 전략 분포 πᵢ
      energy     [N², 1]   내부 에너지 Eᵢ (심신 정동)
      epsilon    [N², 1]   로컬 노이즈 분산 εᵢ
      position   [N², 2]   그리드 좌표
    """

    def __init__(self, cfg: PhinxConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        n_agents = cfg.N * cfg.N

        self.prior    = torch.rand(n_agents, 1,  device=device) * 0.4 + 0.3
        self.strategy = torch.rand(n_agents, 2,  device=device)
        self.strategy = F.normalize(self.strategy, p=1, dim=-1)
        self.energy   = torch.rand(n_agents, 1,  device=device)
        self.epsilon  = torch.rand(n_agents, 1,  device=device) * 0.1 + 0.05

        # 그리드 좌표 생성 [N², 2]
        xs = torch.arange(cfg.N, device=device).float()
        ys = torch.arange(cfg.N, device=device).float()
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.position = torch.stack([
            grid_x.flatten(), grid_y.flatten()
        ], dim=-1)

    def to_qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        에이전트 상태 → Q, K, V 변환
        Q: 현재 상태 (나는 누구인가)
        K: 비교 기준 (상대방이 누구인가)
        V: 전달 정보 (무엇을 교환하는가)
        """
        state = torch.cat([
            self.prior, self.strategy, self.energy, self.epsilon
        ], dim=-1)                                    # [N², D]

        # 패딩으로 D 맞춤
        D = self.cfg.D
        if state.shape[-1] < D:
            pad = D - state.shape[-1]
            state = F.pad(state, (0, pad))
        else:
            state = state[:, :D]

        Q = state
        K = state                                     # 자기참조 어텐션
        V = torch.cat([
            self.prior, self.strategy,
            torch.zeros_like(self.energy),
            self.epsilon
        ], dim=-1)[:, :D]

        return Q, K, V

    @torch.no_grad()
    def apply_attention_update(self, attn_out: torch.Tensor):
        """
        어텐션 출력 → 베이즈 갱신 + 전략 갱신 + ε 공진화
        τ=3 베이즈 갱신을 벡터화하여 GPU에서 처리
        """
        # 어텐션 출력에서 각 성분 추출
        new_prior_signal = attn_out[:, 0:1].sigmoid()   # [0,1] 범위

        # 베이즈 갱신 τ=3 (벡터화)
        p = self.prior
        e = new_prior_signal
        for _ in range(3):
            p = (e * p) / (e * p + (1 - e) * (1 - p) + 1e-8)
        self.prior = p.clamp(0.01, 0.99)

        # 전략 갱신 (게임이론 보수 반영)
        new_strat = attn_out[:, 1:3].abs()
        new_strat = F.normalize(new_strat + 1e-8, p=1, dim=-1)
        self.strategy = 0.85 * self.strategy + 0.15 * new_strat

        # ε 공진화 (로컬 온도 갱신)
        new_eps = attn_out[:, 3:4].abs() * 0.1
        self.epsilon = 0.8 * self.epsilon + 0.2 * new_eps


# ─────────────────────────────────────────────
# 열역학 앙상블 GPU 병렬
# ─────────────────────────────────────────────

class GPUEnsemble:
    """
    M개 샘플을 GPU에서 동시 처리
    분배함수 Z, 엔트로피 S, 유효온도 T*, 자유에너지 F 계산
    """

    def __init__(self, cfg: PhinxConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def compute(self, state: GPUAgentState) -> dict:
        """
        앙상블 추정 — 모든 계산 GPU 텐서 연산
        프레임당 목표: RTX 20xx ~3ms, RTX 30xx ~1.5ms
        """
        M = self.cfg.M
        k = self.cfg.k_boltzmann

        # M개 샘플: ε 노이즈 추가한 prior 샘플링
        eps = torch.randn(
            M, state.prior.shape[0], 1, device=self.device
        ) * state.epsilon.unsqueeze(0)                 # [M, N², 1]

        samples = (state.prior.unsqueeze(0) + eps).clamp(0, 1)  # [M, N², 1]

        # 유효온도 T* = Var(ε) / k  — 로컬 분산에서 창발
        T_star = state.epsilon.var().item() / k
        T_star = max(T_star, 1e-6)

        # 분배함수 Z = Σ exp(-E/kT*)
        E = state.energy                               # [N², 1]
        boltzmann = torch.exp(-E / (k * T_star))
        Z = boltzmann.sum().item()

        # 엔트로피 S = -k Σ p·ln(p)  — 8빈 히스토그램 근사
        prior_flat = state.prior.flatten()
        hist = torch.histc(prior_flat, bins=8, min=0.0, max=1.0)
        hist = hist / hist.sum().clamp(min=1e-8)
        S = -(hist * (hist + 1e-9).log()).sum().item() * k

        # 평균 에너지
        E_mean = E.mean().item()

        # 자유에너지 F = E - T*·S
        F_val = E_mean - T_star * S

        # 협력율 (앙상블 평균)
        coop = samples[:, :, 0].mean().item()

        return {
            'Z': Z,
            'T_star': T_star,
            'S': S,
            'F': F_val,
            'E_mean': E_mean,
            'coop': coop,
        }


# ─────────────────────────────────────────────
# 프랙탈 차원 GPU 계산
# ─────────────────────────────────────────────

@torch.no_grad()
def fractal_dim_gpu(state: GPUAgentState, scales: int = 3) -> float:
    """
    박스카운팅 프랙탈 차원 — GPU 텐서 연산
    2~3 스케일 근사, 실시간 처리 가능
    """
    N = state.cfg.N
    grid = state.prior.reshape(N, N)                  # [N, N]
    binary = (grid > 0.5).float()

    counts = []
    for s in range(scales):
        box = 2 ** s
        if box > N:
            break
        # 박스 집계
        pooled = F.avg_pool2d(
            binary.unsqueeze(0).unsqueeze(0),
            kernel_size=box, stride=box
        ).squeeze()
        counts.append((pooled > 0).float().sum().item())

    if len(counts) < 2 or counts[-1] == 0 or counts[0] == 0:
        return 1.5  # 기본값

    # 로그-로그 선형 회귀 근사 (2점)
    D = math.log(counts[0] / counts[-1]) / math.log(2 ** (scales - 1))
    return max(1.0, min(2.0, D))


import math


# ─────────────────────────────────────────────
# 생존함수 Φ GPU 계산
# ─────────────────────────────────────────────

@torch.no_grad()
def compute_phi_gpu(
    thermo: dict,
    D_fractal: float,
    cfg: PhinxConfig,
) -> dict:
    """
    Φ = sigmoid(α·S + β·D − γ·T*) · ⟨협력율⟩M

    Returns:
        phi          : float [0,1]  생존 지수
        is_critical  : bool         임계점 도달 여부
        + 구성 요소 전체
    """
    S      = thermo['S']
    T_star = thermo['T_star']
    coop   = thermo['coop']

    logit = cfg.alpha * S + cfg.beta * D_fractal - cfg.gamma * T_star
    phi   = (1 / (1 + math.exp(-logit))) * coop

    # 임계 판단: Φ < 0.2 또는 D_fractal < 1.2
    is_critical = (phi < 0.2) or (D_fractal < 1.2)

    return {
        'phi':         phi,
        'S':           S,
        'D':           D_fractal,
        'T_star':      T_star,
        'coop':        coop,
        'F':           thermo['F'],
        'is_critical': is_critical,
    }


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────

class PhinxPipeline:
    """
    phinx 전체 GPU 파이프라인

    사용법:
        cfg = PhinxConfig(N=32, M=64)
        cfg.auto_tune()  # GPU에 맞게 자동 조정

        pipeline = PhinxPipeline(cfg)

        for frame in pipeline.run():
            # frame['phi']       → Φ 생존 지수
            # frame['S']         → 엔트로피 (색 다양성 매핑)
            # frame['D']         → 프랙탈 차원 (패턴 복잡성 매핑)
            # frame['T_star']    → 유효온도 (사운드 거칠기 매핑)
            # frame['is_critical'] → 임계 도달 여부
            send_to_output(frame)
    """

    def __init__(self, cfg: PhinxConfig):
        self.cfg = cfg
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.state    = GPUAgentState(cfg, self.device)
        self.ensemble = GPUEnsemble(cfg, self.device)

        cc = _get_cc()
        print(f"[phinx] 파이프라인 초기화 완료")
        print(f"[phinx] Device: {self.device} | CC: {cc[0]}.{cc[1]}")
        print(f"[phinx] 커널: {'Triton' if cc >= (7,5) and torch.cuda.is_available() else 'Torch'}")

    @torch.no_grad()
    def step(self) -> dict:
        """
        프레임 1회 실행
        목표: RTX 20xx ~12ms, RTX 30xx ~6ms
        """
        # 1. 에이전트 상태 → Q, K, V
        Q, K, V = self.state.to_qkv()

        # 2. 어텐션 커널 (이웃 마스크 적용)
        #    빗방울 충돌 + 베이즈 갱신을 GPU에서 한 번에
        attn_out = masked_agent_attention(
            Q, K, V,
            positions=self.state.position,
            r=self.cfg.r,
        )

        # 3. 상태 갱신
        self.state.apply_attention_update(attn_out)

        # 4. 열역학 앙상블 추정
        thermo = self.ensemble.compute(self.state)

        # 5. 프랙탈 차원
        D_fractal = fractal_dim_gpu(self.state)

        # 6. 생존함수 Φ
        result = compute_phi_gpu(thermo, D_fractal, self.cfg)

        return result

    def run(self, max_frames: int | None = None):
        """프레임 제너레이터 — 실시간 루프"""
        import time
        frame_budget = 1.0 / self.cfg.fps_target

        frame = 0
        while max_frames is None or frame < max_frames:
            t0 = time.perf_counter()
            result = self.step()
            result['frame'] = frame

            elapsed = time.perf_counter() - t0
            result['ms'] = elapsed * 1000

            yield result

            # 남은 시간 대기 (fps 조절)
            remaining = frame_budget - elapsed
            if remaining > 0:
                time.sleep(remaining)

            frame += 1


# ─────────────────────────────────────────────
# 빠른 확인용
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import time

    cfg = PhinxConfig(N=32, M=64, D=16)
    cfg.auto_tune()

    pipeline = PhinxPipeline(cfg)

    print("\n--- 10 프레임 벤치마크 ---")
    times = []
    for result in pipeline.run(max_frames=10):
        times.append(result['ms'])
        status = "⚠ CRITICAL" if result['is_critical'] else "✓"
        print(
            f"Frame {result['frame']:3d} | "
            f"Φ={result['phi']:.3f} | "
            f"S={result['S']:.3f} | "
            f"D={result['D']:.3f} | "
            f"T*={result['T_star']:.4f} | "
            f"{result['ms']:.1f}ms {status}"
        )

    print(f"\n평균: {sum(times)/len(times):.1f}ms | "
          f"최대: {max(times):.1f}ms | "
          f"60fps 기준(16ms): {'✓' if max(times) < 16 else '△ 조정 필요'}")
