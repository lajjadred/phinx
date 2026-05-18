# phinx — Installation & Quick Start

## 1. Install Dependencies

Run inside your venv:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install triton
pip install numpy scipy python-osc websockets
```

> Triton is often bundled with torch automatically.
> Verify with `pip show triton` after installation.

**CPU-only environment (no GPU):**
```bash
pip install torch torchvision
pip install numpy scipy python-osc websockets
```
Triton is not required — the pipeline falls back to PyTorch automatically.

---

## 2. File Structure

```
phinx/
└── phinx/
    └── core/
        ├── __init__.py          (empty file)
        ├── attention_kernel.py
        └── gpu_pipeline.py
```

---

## 3. Verify Installation

```bash
# Test attention kernel only
python phinx\core\attention_kernel.py

# Test full pipeline (10-frame benchmark)
python phinx\core\gpu_pipeline.py
```

Expected output:
```
[phinx] GPU: CC 8.6, VRAM 12.0GB
[phinx] Auto config → N=32, M=128, D=16
[phinx] Kernel: Triton

--- 10 Frame Benchmark ---
Frame   0 | Φ=0.412 | S=1.823 | D=1.641 | T*=0.0031 | 6.2ms ✓
Frame   1 | Φ=0.398 | S=1.791 | D=1.658 | T*=0.0029 | 5.8ms ✓
...
Average: 6.1ms | Max: 7.3ms | 60fps target (16ms): ✓
```

---

## 4. Expected Performance by GPU

| GPU           | CC  | Per-frame  | Recommended M |
|---|---|---|---|
| CPU only      | —   | ~40–80ms  | 32            |
| RTX 2060/70   | 7.5 | ~12ms     | 64            |
| RTX 2080 Ti   | 7.5 | ~8ms      | 128           |
| RTX 3060      | 8.6 | ~6ms      | 128           |
| RTX 3080/90   | 8.6 | ~3ms      | 512           |

All GPU targets are within the 16ms (60fps) budget.

If running CPU-only and 60fps is required, set `PhinxConfig(N=16, M=32)`.

---

## 5. Kernel Dispatch Logic

The pipeline selects the kernel automatically at runtime:

```
GPU available + CC ≥ 7.5 + Triton installed  →  Triton kernel
otherwise                                     →  PyTorch fallback
```

No code changes needed — works on any hardware.

---

## 6. Output Values → Artwork Mapping

| Variable      | Meaning               | Example Mapping                  |
|---|---|---|
| `phi`         | Survival index        | Global brightness, volume        |
| `S`           | Entropy               | Color diversity, particle spread |
| `D`           | Fractal dimension     | Pattern complexity, texture      |
| `T_star`      | Effective temperature | Sound roughness, frequency       |
| `coop`        | Cooperation rate      | Center density                   |
| `is_critical` | Phase transition flag | Collapse event trigger           |

---

## 7. Real-time Output via OSC

Connect to TouchDesigner, Max/MSP, or SuperCollider:

```python
from phinx.core.gpu_pipeline import PhinxPipeline, PhinxConfig
from pythonosc import udp_client

cfg = PhinxConfig()
cfg.auto_tune()

pipeline = PhinxPipeline(cfg)
osc = udp_client.SimpleUDPClient("127.0.0.1", 9000)

for frame in pipeline.run():
    osc.send_message("/phinx/phi",      frame['phi'])
    osc.send_message("/phinx/entropy",  frame['S'])
    osc.send_message("/phinx/fractal",  frame['D'])
    osc.send_message("/phinx/temp",     frame['T_star'])
    osc.send_message("/phinx/critical", int(frame['is_critical']))
```

OSC message schema:

```
/phinx/phi        f   [0.0, 1.0]   survival index
/phinx/entropy    f   [0.0, ∞)     diversity
/phinx/fractal    f   [1.0, 2.0]   pattern complexity
/phinx/temp       f   [0.0, ∞)     effective temperature
/phinx/critical   i   [0, 1]       phase transition flag
```

---

## 8. Minimal Usage Example

```python
from phinx.core.gpu_pipeline import PhinxPipeline, PhinxConfig

cfg = PhinxConfig(N=32, M=64)
cfg.auto_tune()

pipeline = PhinxPipeline(cfg)

for frame in pipeline.run():
    print(f"Φ={frame['phi']:.3f}  {frame['ms']:.1f}ms")
```

---

## 9. License

MIT — free for academic, artistic, and commercial use.
