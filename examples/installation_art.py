"""
examples/installation_art.py
-----------------------------
NEMAF 설치 작품 기본 예제.

관람객 위치 시뮬레이션 → 그리드 갱신 → Φ 계산 → 콘솔 출력.
실제 작품에서는 ConsoleOutput을 OSC/WebSocket으로 교체.

실행:
    python examples/installation_art.py
"""

import numpy as np
import phinx
from phinx.output.realtime import ConsoleOutput


def simulate_visitors(grid, n_visitors: int = 5):
    """
    관람객 위치 시뮬레이션.
    실제 작품: 카메라/LiDAR 센서 데이터로 교체.
    """
    N = grid.N
    for _ in range(n_visitors):
        # 랜덤 위치
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        # 관람객 관여도 → prior 갱신
        engagement = float(np.clip(np.random.rand(), 0.3, 0.9))
        grid.agents[i][j].prior = engagement
        # 체류 시간 → epsilon 감소 (더 확신)
        grid.agents[i][j].epsilon_var *= 0.95


def main():
    print("=" * 60)
    print("phinx — NEMAF 설치 작품 시뮬레이션")
    print("=" * 60)

    # 초기화
    np.random.seed(None)  # 매 실행마다 다른 시작

    grid = phinx.EnsembleGrid(N=16, r=1, wrap=True)

    # 초기 prior 랜덤 설정
    for i in range(grid.N):
        for j in range(grid.N):
            grid.agents[i][j].prior = float(
                np.clip(np.random.rand(), 1e-6, 1 - 1e-6)
            )

    ensemble = phinx.ThermoEnsemble(grid, M=64)
    console  = ConsoleOutput(every_n=5)

    # PhiLoop으로 100프레임 실행
    loop = phinx.PhiLoop(
        grid, ensemble,
        fps=30,
        alpha=0.3,  # 다양성(S) 가중치
        beta=0.4,   # 복잡성(D) 가중치
        gamma=0.3,  # 혼돈(T*) 가중치
    )

    def on_frame(result):
        # 관람객 시뮬레이션 (10프레임마다)
        if result["frame"] % 10 == 0:
            simulate_visitors(grid, n_visitors=3)

        # 콘솔 출력
        console.send(result)

        # 상전이 경보
        if result["signal"] == "critical":
            print(f"\n  ⚠️  상전이 임계 감지! frame={result['frame']} "
                  f"Φ={result['phi']:.3f}\n")

    print("\n시뮬레이션 시작 (100프레임)...\n")
    results = loop.run(n_frames=100, callback=on_frame)

    # 최종 요약
    summary = loop.summary()
    print("\n" + "=" * 60)
    print("최종 요약")
    print("=" * 60)
    print(f"  총 프레임    : {summary['n_frames']}")
    print(f"  Φ 평균       : {summary['phi_mean']:.3f}")
    print(f"  Φ 최솟값     : {summary['phi_min']:.3f}")
    print(f"  Φ 최댓값     : {summary['phi_max']:.3f}")
    print(f"  Φ 최종값     : {summary['phi_final']:.3f}")
    print(f"  평균 처리시간 : {summary['avg_total_ms']:.1f}ms/frame")
    print(f"  최대 처리시간 : {summary['max_total_ms']:.1f}ms/frame")

    signals = summary["signals"]
    from collections import Counter
    sc = Counter(signals)
    print(f"\n  신호 분포:")
    print(f"    🟢 stable   : {sc.get('stable', 0)}회")
    print(f"    🟡 warning  : {sc.get('warning', 0)}회")
    print(f"    🔴 critical : {sc.get('critical', 0)}회")
    print()


if __name__ == "__main__":
    main()
