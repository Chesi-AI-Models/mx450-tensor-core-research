# D1 GEMM optimisation + D2 clock-locking experiment — complete measurement report

Date: 2026-10-20
GPU: NVIDIA GeForce MX450 (TU117), 14 SM, CC 7.5, VRAM 2 GB
Driver: nvidia-open 580.142 (GSP forcibly enabled)
Test scale: N=2048, FP16 input / FP32 accumulation, FLOP = 2·N³ = 1.718e10

---

## 1. Summary of conclusions

| Metric | Result |
|---|---|
| best self-built FFMA GEMM | **2.4872 TFLOPS** (C7 `64×128×BK8 T8×8 PAD0 single-buffer`) |
| vs cuBLAS fp32 mm (measured in the same process) | **1.17×** (cuBLAS 2.13 TFLOPS) |
| vs cuBLAS fp16 mm | **8.1×** (cuBLAS fp16 only 0.32 TFLOPS) |
| clock-normalised efficiency | **80–81%** (compare: pure FP32 FFMA reaches 99.0%) |
| `nvidia-smi -lgc` clock lock | **accepted (rc=0) but completely ineffective** |
| thermal throttle flag | **Not Active** (so it is not thermally suppressed) |

**★ The most important correction: the "3.06 TFLOPS" denominator is wrong.**
It came from an assumed 1665 MHz, but this machine's SM clock drifts between **1290–1830 MHz**
(`clocks.max.sm = 2100`). Therefore:

- physical peak = `2 × 64 × 14 × clk` = 1290MHz→2.31 / 1395→2.50 / 1665→2.98 / 1830→3.28 / 2100→3.76 TFLOPS
- **absolute TFLOPS are not comparable across runs**; only the percentage relative to the same-run clock-normalised
  peak is comparable
- the lesson is the same class as T2-1's "MB/GB unit mixing": **if the denominator is not calibrated, every
  percentage is fake**

---

## 2. The D1 optimisation process (every step is measured data)

| Version | Change | TFLOPS | Clock-normalised efficiency | Conclusion |
|---|---|---|---|---|
| v2 | scalar smem, BK=8 PAD=6, 128×128×256 | 1.603 | 66.3% | baseline |
| v4 | A tile transposed + BK×PAD sweep | 1.876 | 77.6% | the transpose idea disproven (16-bit write conflicts > read gains), but the BK/PAD sweep was useful |
| v5 | **float4/uint4 global loads** (16 B) | 1.9714 | 81.6% | **effective**; best A:v4 B:v4 BK=16 PAD=8 |
| v6 | float smem + A transposed + inner LDS.128 | 1.811 | — | **regression**: smem bytes double + transposed scalar write conflicts |
| v7/v8 | tile × threads × double-buffer sweep | 2.3075→2.4872 | ~81% | **double buffering effective**; BK=8 beats BK=16 |
| round C (calibration) | separate process + cuBLAS calibration point | **2.4872** | 80.9% | see table below |

### 2.1 Calibrated configuration comparison (two independent sweeps, reproducible numbers)

| Configuration | best TFLOPS (run 1 / run 2) | regs | blocks/SM |
|---|---|---|---|
| C1 128×128×8 P8 single | 2.2157 / 2.1472 | 122 | 2 |
| C2 128×128×8 P8 **double** | 2.4407 / 2.4578 | 128 | 2 |
| C3 128×128×16 P8 single | 2.3527 / 2.2902 | 124 | 2 |
| C4 128×128×16 P8 double | 2.3977 / 2.3295 | 125 | 2 |
| C5 128×64×16 P8 single | 2.4333 / 2.3953 | 125 | 4 |
| C6 128×64×16 P8 double | 2.3459 / 2.2580 | 127 | 4 |
| **C7 64×128×8 P0 single** | **2.4872 / 2.4755** | 122 | 4 |
| **C8 128×128×8 P16 double** | 2.4851 / 2.4766 | 128 | 2 |
| calibration cuBLAS fp32 | 2.1333 / 2.1471 | — | — |
| calibration cuBLAS fp16 | 0.3191 / 0.3171 | — | — |

Same-run drift is only 1.02–1.09×, showing that **separate process + interleaving + cuBLAS calibration** is a
reliable method.

### 2.2 Key mechanism verifications

**① float2 ≠ 2 halves (a pitfall)**
CUDA's `float2` is 2 **floats** = 8 bytes. To represent 2 halves (4 bytes) you must use
`unsigned int`. Using the wrong type immediately triggers `misaligned address`, and this is a **sticky error** that
kills the whole context, causing every subsequent kernel to report 716/700. → you must write a **static alignment
pre-check**, not rely on crashing to find out.

**② padding breaks 16 B alignment (confirmed by web search + measurement)**
For the A tile `[BM][BK+PAD]`, to write shared memory with float4 (16 B) you need `(BK+PAD)×2 % 16 == 0`,
i.e. `BK+PAD` must be a multiple of 8. With BK=8, PAD=6 → row start offset 12 B, caught immediately by the static
pre-check.
The valid PAD values are {0, 8, 16}. **With PAD=0 the row stride is exactly 16 B, giving naturally zero bank
conflicts** (C7 uses PAD=0).

**③ Reducing inner-loop instruction count does not mean getting faster**
v6 changed the inner loop from `16×LDS.U16 + 16×cvt` to `4×LDS.128`, cutting instruction count to 1/4,
but the overall result was **8% slower**. The reason is that storing floats in smem doubles the byte volume, and the
transposed writes are 8 scalar STSs.
→ "instruction count" is not the only dimension; **smem byte traffic** matters equally.

**④ The pure-FFMA ceiling is trustworthy (99.0%)**
`2.4747 TFLOPS @ 1395 MHz`, denominator `2×64×14×1.395 = 2.4998` → 99.0%.
This shows the formula and the FMA pipeline can both be saturated; GEMM's 80.9% is short by **memory-access stalls**,
not compute.

---

## 3. D2 clock-locking / overclocking experiments (all on the real machine, root)

| Means | Command | Result |
|---|---|---|
| lock SM clock 1500 | `nvidia-smi -lgc 1500` | rc=0 "GPU clocks set", but **measured still 1320** |
| lock 1650 / 1800 / 1950 / 2100 | same | rc=0, measured 1305 / 1290 / 1290 / 1290, and **GEMM dropped to 1.85 instead** |
| power limit | `nvidia-smi -pl 35` | **Changing power management limit is not supported** |
| thermal throttle flag | `clocks_event_reasons.active` | **0x0000000000000000**; hw/sw thermal slowdown all Not Active |
| power readings | `power.draw` / `power.limit` | **[N/A] / [N/A]** (default/max report 5001 W, an invalid value) |
| fan speed | `fan.speed` + `/sys/class/drm/card*/device/hwmon/*/pwm*` | **[N/A], no pwm node** |

### 3.1 Why `-lgc` is ineffective (inference with existing supporting evidence)

- the command is **accepted by the driver and echoes success**, yet the clock does not move at all → not a permission
  issue, not a syntax issue
- **all thermal throttle flags are 0** → not thermally suppressed (temperature 80–86 °C, but no throttle flag)
- this project has already measured and confirmed: **the open module forcibly enables GSP** (`NVreg_EnableGpuFirmware=0`
  is ignored)
- the GSP-RM firmware has taken over the clock/power domains → `nvidia-smi`'s lgc goes through the old PMU path,
  which is bypassed

**→ to genuinely overclock there are only two routes:**
1. **switch to the closed-source module** + `NVreg_EnableGpuFirmware=0`, returning clock management to the driver side
   (open-gpu-kernel-modules issue #965 has a success case, at the cost of 50% lower performance in some scenarios) —
   not yet done
2. modify the vBIOS/PPTable — **this machine has no vBIOS** (ROM BAR allocation fails, already measured), so this
   route is closed

---

## 4. File inventory

| File | Purpose |
|---|---|
| `sw_tensor_core_v5_vec.py` | float2/float4 global loads + static alignment pre-check |
| `sw_tensor_core_v6_fp32smem.py` | float smem + A transposed + inner LDS.128 (disproven) |
| `sw_tensor_core_v7_sweep.py` | first tile × threads × double-buffer sweep |
| `sw_tensor_core_v8_around_s5.py` | fine sweep around S5 (PAD/BK/tile) |
| `d1_diag_ceiling.py` | register / occupancy diagnostics |
| `d1_bench_calibrated.py` | **separate process + cuBLAS calibration point** (recommended benchmarking tool) |
| `d1_peak_truth.py` | pure FP32/FP64 FMA peak + clock back-solve |
| `d1_clock_during.py` | clock sampling while the kernel runs |
| `d2_clock_lock.py` | clock-lock / power-limit experiments (needs root) |

---

## 5. Next steps

1. **the remaining 8–18% of D1**: target raising clock-normalised efficiency from 81% to ≥90% (matching cuBLAS fp32's
   89.6%). Directions: 3-stage pipelining (prefetch two blocks), fewer `__syncthreads`, splitting cvt into the load stage
2. **the only feasible D2 route**: switch to the closed-source module to verify whether `-lgc` takes effect
   (reversible; recovery steps must be recorded)
3. all future benchmarks **must** use `d1_bench_calibrated.py`'s method (separate process + same-run cuBLAS calibration)
