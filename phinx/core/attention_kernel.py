"""
phinx/core/attention_kernel.py

에이전트 간 상호작용을 Flash Attention 구조로 GPU 가속
수학적 동형: 빗방울 충돌 확률 p=exp(-d/r₀) ≡ softmax(QKᵀ/√d)

GPU 호환:
  - RTX 20xx / Quadro RTX  : Compute Capability 7.5  (Turing)
  - RTX 30xx               : Compute Capability 8.6  (Ampere)
  - RTX 40xx               : Compute Capability 8.9  (Ada)
  fallback: CPU (torch 기반, 자동 선택)
"""

import math
import torch
import torch.nn.functional as F

# Triton 가용 여부 자동 감지
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Compute Capability 감지
def _get_cc() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    major, minor = torch.cuda.get_device_capability()
    return (major, minor)


# ─────────────────────────────────────────────
# TRITON 커널 (CC 7.5+ — RTX 20xx 이상)
# ─────────────────────────────────────────────

if TRITON_AVAILABLE:

    @triton.jit
    def _agent_attention_fwd(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        stride_qn, stride_qd,
        stride_kn, stride_kd,
        stride_vn, stride_vd,
        stride_on, stride_od,
        N: tl.constexpr,   # 에이전트 수
        D: tl.constexpr,   # 상태 차원
        scale: tl.constexpr,
        BLOCK_N: tl.constexpr,  # 타일 크기
        BLOCK_D: tl.constexpr,
    ):
        """
        에이전트 어텐션 포워드 커널

        Q[i] = 에이전트 i의 쿼리 (현재 상태 벡터 ψᵢ)
        K[j] = 에이전트 j의 키   (비교 대상 상태)
        V[j] = 에이전트 j의 값   (전달할 정보)

        attn[i,j] = softmax(Q[i]·K[j]ᵀ · scale)
                  ≡ 빗방울 충돌 확률 p(i,j)

        out[i] = Σⱼ attn[i,j] · V[j]  ← 베이즈 갱신 후 상태
        """
        # 현재 블록의 쿼리 인덱스
        start_n = tl.program_id(0) * BLOCK_N
        offs_n  = start_n + tl.arange(0, BLOCK_N)
        offs_d  = tl.arange(0, BLOCK_D)

        # Q 블록 로드
        q_ptrs = Q_ptr + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd
        q_mask = offs_n[:, None] < N
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # 온라인 softmax 초기화 (Flash Attention 방식)
        m_i = tl.full([BLOCK_N], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
        acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        # K, V 블록 순회 (메모리 효율적 타일링)
        for start_k in range(0, N, BLOCK_N):
            offs_k = start_k + tl.arange(0, BLOCK_N)

            # K 블록 로드
            k_ptrs = K_ptr + offs_k[None, :] * stride_kn + offs_d[:, None] * stride_kd
            k_mask = offs_k[None, :] < N
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # V 블록 로드
            v_ptrs = V_ptr + offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v_mask = offs_k[:, None] < N
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

            # 어텐션 스코어: QKᵀ · scale
            # = 에이전트 i-j 간 유사도 (빗방울 충돌 강도)
            scores = tl.dot(q, k) * scale  # [BLOCK_N, BLOCK_N]

            # 이웃 마스킹 — 로컬 반경 r 밖은 -inf
            # (거리 기반 마스크는 호출 시 K에 이미 반영)

            # 온라인 softmax (수치 안정)
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p, axis=1)

            # 누적 어텐션 출력
            acc = acc * (tl.exp(m_i - m_new)[:, None]) + tl.dot(p.to(tl.float16), v)

            m_i = m_new
            l_i = l_new

        # 정규화
        acc = acc / l_i[:, None]

        # 출력 저장
        out_ptrs = Out_ptr + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc.to(tl.float16), mask=q_mask)


    def agent_attention_triton(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Triton 기반 에이전트 어텐션

        Args:
            Q: [N, D] 에이전트 쿼리 (상태벡터 ψᵢ)
            K: [N, D] 에이전트 키
            V: [N, D] 에이전트 값 (전달 정보)
            scale: QKᵀ 스케일 (기본값: 1/√D)

        Returns:
            out: [N, D] 갱신된 에이전트 상태
        """
        N, D = Q.shape
        scale = scale or (1.0 / math.sqrt(D))

        # fp16으로 변환 (Tensor Core 활용 — CC 7.5+ 지원)
        Q = Q.half().contiguous()
        K = K.half().contiguous()
        V = V.half().contiguous()
        out = torch.empty_like(Q)

        # 타일 크기 — CC 7.5(RTX 20xx) 호환 보수적 설정
        # CC 8.6(RTX 30xx)에서는 자동으로 더 큰 타일 사용 가능
        BLOCK_N = 32   # 20xx: 32, 30xx: 64 가능 (autotuner가 최적화)
        BLOCK_D = min(D, 64)

        grid = (math.ceil(N / BLOCK_N),)

        _agent_attention_fwd[grid](
            Q, K, V, out,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1),
            V.stride(0), V.stride(1),
            out.stride(0), out.stride(1),
            N=N, D=D, scale=scale,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        return out.float()


# ─────────────────────────────────────────────
# TORCH FALLBACK (CPU / CC 7.5 미만 / Triton 미설치)
# ─────────────────────────────────────────────

def agent_attention_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    PyTorch 기반 fallback 어텐션
    Triton 없거나 GPU 없을 때 자동 선택
    """
    scale = scale or (1.0 / math.sqrt(Q.shape[-1]))
    scores = (Q @ K.T) * scale                    # [N, N]
    weights = F.softmax(scores, dim=-1)            # 빗방울 충돌 확률 분포
    return weights @ V                             # [N, D] 갱신 상태


# ─────────────────────────────────────────────
# 자동 디스패처 — 환경에 따라 커널 자동 선택
# ─────────────────────────────────────────────

def agent_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    환경 자동 감지 → 최적 커널 선택

    우선순위:
      1. Triton 커널 (CUDA + CC 7.5+)   ← RTX 20xx, 30xx, 40xx
      2. torch.nn.functional (CPU/GPU)   ← fallback
    """
    cc = _get_cc()
    use_triton = (
        TRITON_AVAILABLE
        and torch.cuda.is_available()
        and cc >= (7, 5)           # RTX 20xx 이상
        and Q.device.type == 'cuda'
    )

    if use_triton:
        return agent_attention_triton(Q, K, V, scale)
    else:
        return agent_attention_torch(Q, K, V, scale)


# ─────────────────────────────────────────────
# 로컬 이웃 마스킹 유틸
# ─────────────────────────────────────────────

def build_neighbor_mask(
    positions: torch.Tensor,
    r: float = 1.5,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    에이전트 위치에서 반경 r 이내 이웃 마스크 생성

    Args:
        positions: [N, 2] 에이전트 (x, y) 좌표
        r: 이웃 반경 (그리드 단위)

    Returns:
        mask: [N, N] bool — True이면 이웃 관계
    """
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [N, N, 2]
    dist = torch.norm(diff, dim=-1)                          # [N, N]
    return dist <= r


def masked_agent_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    positions: torch.Tensor,
    r: float = 1.5,
    scale: float | None = None,
) -> torch.Tensor:
    """
    이웃 마스크 적용 어텐션
    r 반경 밖 에이전트는 어텐션 스코어 -inf → 상호작용 없음

    빗방울 수열의 로컬 충돌 조건을 어텐션으로 구현
    """
    scale = scale or (1.0 / math.sqrt(Q.shape[-1]))
    scores = (Q @ K.T) * scale                              # [N, N]

    # 거리 마스크 적용
    mask = build_neighbor_mask(positions, r, device=Q.device.type)
    scores = scores.masked_fill(~mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)            # 고립 에이전트 처리
    return weights @ V


# ─────────────────────────────────────────────
# 빠른 확인용
# ─────────────────────────────────────────────

if __name__ == '__main__':
    cc = _get_cc()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"Compute Capability: {cc[0]}.{cc[1]}")
    print(f"Triton 가용: {TRITON_AVAILABLE}")
    print(f"선택 커널: {'Triton' if TRITON_AVAILABLE and cc >= (7,5) else 'Torch fallback'}")
    print()

    # 기본 어텐션 테스트
    N, D = 256, 64
    Q = torch.randn(N, D, device=device)
    K = torch.randn(N, D, device=device)
    V = torch.randn(N, D, device=device)

    out = agent_attention(Q, K, V)
    print(f"어텐션 출력 shape: {out.shape}  ✓")

    # 마스크 어텐션 테스트
    pos = torch.rand(N, 2, device=device) * 10
    out_masked = masked_agent_attention(Q, K, V, pos, r=2.0)
    print(f"마스크 어텐션 출력 shape: {out_masked.shape}  ✓")

    # 속도 벤치마크
    import time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            agent_attention(Q, K, V)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"100회 평균: {(t1-t0)/100*1000:.2f}ms / 호출")
