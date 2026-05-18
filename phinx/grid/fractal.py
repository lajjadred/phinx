"""
phinx.grid.fractal
------------------
프랙탈 차원 계산 — 박스카운팅 근사.

실시간 처리 전제: 2~4 스케일만 사용 → O(N²) 경량 계산.
D ∈ [1.0, 2.0] — 1에 가까울수록 단순, 2에 가까울수록 복잡.
정상 복잡계 D ≈ 1.6~1.8 / 급락 시 상전이 신호.
"""

from __future__ import annotations

import numpy as np


def fractal_dim_boxcount(
    matrix: np.ndarray,
    threshold: float = 0.5,
    scales: int = 3,
) -> float:
    """
    박스카운팅 프랙탈 차원.

    Parameters
    ----------
    matrix    : ndarray (N, N)  상태 행렬 [0,1]
    threshold : float           이진화 임계값
    scales    : int             사용할 박스 스케일 수 (2~4 권장)

    Returns
    -------
    float : 프랙탈 차원 D ∈ [1.0, 2.0]
            계산 불가 시 1.5 반환 (중립값)
    """
    binary = (matrix > threshold).astype(np.float64)
    N = binary.shape[0]

    counts = []
    box_sizes = []

    for k in range(scales):
        box = 2 ** k  # 1, 2, 4, 8, ...
        if box >= N:
            break
        # 박스 크기로 다운샘플 → 박스 내 1이 있으면 카운트
        trimN = (N // box) * box
        reshaped = binary[:trimN, :trimN].reshape(
            N // box, box, N // box, box
        )
        count = (reshaped.max(axis=(1, 3)) > 0).sum()
        if count > 0:
            counts.append(count)
            box_sizes.append(box)

    if len(counts) < 2:
        return 1.5  # 계산 불가 — 중립값

    # 선형 회귀: log(count) ~ D * log(1/box)
    log_inv_box = np.log(1.0 / np.array(box_sizes, dtype=float))
    log_count   = np.log(np.array(counts, dtype=float))

    # 최소제곱 기울기
    A = np.vstack([log_inv_box, np.ones(len(log_inv_box))]).T
    result = np.linalg.lstsq(A, log_count, rcond=None)
    D = float(result[0][0])

    return float(np.clip(D, 1.0, 2.0))


def fractal_dim_multiscale(
    matrix: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    다중 스케일 프랙탈 분석 — 진단용.

    Returns
    -------
    dict:
        D_2scale : 2스케일 근사 (실시간용)
        D_3scale : 3스케일 근사
        D_4scale : 4스케일 근사
        is_healthy : bool  D ∈ [1.5, 1.9] 이면 True
    """
    return {
        "D_2scale":   fractal_dim_boxcount(matrix, threshold, scales=2),
        "D_3scale":   fractal_dim_boxcount(matrix, threshold, scales=3),
        "D_4scale":   fractal_dim_boxcount(matrix, threshold, scales=4),
        "is_healthy": 1.5 <= fractal_dim_boxcount(matrix, threshold, scales=3) <= 1.9,
    }


def fractal_dim_history(history: list[float],
                        window: int = 10) -> dict:
    """
    프랙탈 차원 시계열에서 급락 감지.

    Parameters
    ----------
    history : list[float]  프레임별 D 값 시계열
    window  : int          비교 윈도우 크기

    Returns
    -------
    dict:
        current  : 현재 D
        baseline : 윈도우 평균
        drop     : 급락량 (baseline - current)
        alert    : bool  drop > 0.2 이면 상전이 경보
    """
    if len(history) < window + 1:
        return {"current": history[-1] if history else 1.5,
                "baseline": 1.5, "drop": 0.0, "alert": False}

    current  = history[-1]
    baseline = float(np.mean(history[-window-1:-1]))
    drop     = baseline - current

    return {
        "current":  current,
        "baseline": baseline,
        "drop":     drop,
        "alert":    drop > 0.2,
    }
