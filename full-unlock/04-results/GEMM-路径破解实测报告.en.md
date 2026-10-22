# MX450 (TU117) GEMM path cracking — measurement report

> All data from the real LYLNAS machine (NVIDIA GeForce MX450 / TU117 / CC 7.5 / 14 SM / 1.64 GiB)
> Environment: torch 2.3.1+cu121, CUDA 12.1, driver nvidia-open 580.142
> Date: 2026-10-19

---

## 0. One-line conclusion

**Never do GEMM directly in fp16 on an MX450.**
For CC 7.5, cuBLAS chooses `mma.sync.m8n8k4` optimised for sm_70 (SASS: `HMMA.884`);
TU117 has no hardware that can execute it and can only emulate — measured **7.5× slower**.
Switching to an fp32 intermediate or blocking immediately recovers **3.3× ~ 4.5×**, and the fp32 route has
**zero precision loss**.

---

## 1. Root-cause forensics: the kernel name settles it

Capturing the actual kernels at N=1024 with `torch.profiler`:

| dtype | Measured time | Kernel name |
|---|---|---|
| **fp16** | **20318 us** | `cutlass::Kernel<cutlass_70_tensorop_f16_s884gemm_relu_f16_128x64_nn_align8>` |
| **fp32** | **2685 us** | `volta_sgemm_128x64_nn` |
| **bf16** | 4239 us | `magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, ...>` |

**What `s884` means**: `mma.sync.aligned.m8n8k4`, a Volta-era warp-level MMA instruction.
The SASS mapping table states plainly: `mma.m8n8k4.f16 (Volta form) → HMMA.884.F32.F32.STEP{0..3}`, scope **sm_75**.

**NVIDIA's own PTX documentation, verbatim**:

> Note: `mma.sync.m8n8k4` is optimized for target architecture `sm_70` and **may have
> substantially reduced performance on other target architectures**.

On CC 7.5, cuBLAS picks exactly the instruction the vendor warns about. That is the entire source of the 7.5× loss.

---

## 2. Three independent corroborations (not a quirk of our one machine)

| Source | Hardware | Conclusion |
|---|---|---|
| this report (real machine) | MX450 / TU117 | fp16 is **7.45×** slower than fp32 (N=1024) |
| GitHub `msraaghavan/HumanDiT` hardware_notes.md | **GTX 1650 Ti / TU117** | `GEMM 4096² fp32 1.72 TFLOPS, fp16 0.41 TFLOPS (4× slower: no tensor cores, no fast HGEMM path)`; conclusion **"precision policy uses fp32 on sm_75"**; `bf16 emulated; not useful`; FlashAttention requires sm_80+ |
| GitHub `jamesburton/dotLLM` GPU.md | general | "On pre-Volta GPUs cuBLAS FP16 GEMM falls back to CUDA cores" |
| CSDN Q&A | MX450 | "FP16 actually takes the FP32-unit truncation emulation path… sustained throughput drops to about **0.25 TFLOPS**" |

Our measured fp16 across sizes is **0.239 ~ 0.351 TFLOPS**, matching CSDN's "0.25 TFLOPS".

---

## 3. Size sweep: fp32's sweet spot and cliff

| N | fp16 TFLOPS | fp16 ms | fp32 TFLOPS | fp32 ms | fp32/fp16 |
|---|---|---|---|---|---|
| 256 | 0.2389 | 0.140 | 0.5993 | 0.056 | 2.51× |
| 512 | 0.2472 | 1.086 | 1.3148 | 0.204 | 5.32× |
| **1024** | 0.3183 | 6.746 | **2.3714** | 0.906 | **7.45×** |
| **2048** | 0.3268 | 52.575 | **2.1098** | 8.143 | **6.46×** |
| 3072 | 0.3514 | 165.022 | 0.9330 | 62.145 | 2.66× |
| 4096 | 0.3490 | 393.815 | **0.3533** | 388.974 | 1.01× |

**Two key facts:**
1. fp16's TFLOPS barely grows with size (0.24 → 0.35), showing the bottleneck is fixed in instruction emulation,
   not scheduling/occupancy
2. **fp32 is in its sweet spot at 1024/2048 (2.1–2.4 TFLOPS, 77% of the physical peak 3.06), halves at 3072,
   and falls to 0.35 at 4096 (a cliff)**

bf16 for comparison: 1024 → 1.356 TFLOPS, 2048 → 1.299. **4.3× faster than fp16**,
because cuBLAS uses MAGMA `sgemmEx` to promote bf16 to fp32 for computation — which incidentally avoids the bad path.

---

## 4. Option one: fp16 → fp32 → GEMM → fp16 (lossless)

```python
def mx450_mm_fp16(a, b):          # a, b: fp16 CUDA tensors
    return torch.mm(a.float(), b.float()).half()
```

| N | fp16 direct | **fp16→fp32→mm→fp16** | pure fp32 | speedup | max_abs_err |
|---|---|---|---|---|---|
| 1024 | 6.760 ms | **1.828 ms** | 1.621 ms | **3.70×** | **0.000000** |
| 2048 | 54.014 ms | **16.199 ms** | 15.778 ms | **3.33×** | **0.000000** |
| 3072 | 167.870 ms | 98.366 ms | 97.255 ms | 1.71× | 0.000000 |
| 4096 | 398.797 ms | 628.999 ms | 628.690 ms | **0.63×** ⚠ | 0.000000 |

**Note**: 4096 actually gets slower, because fp32 itself falls off the cliff there.
**At 4096, option two (blocking) must be combined.**

### Alternative: fp16 → bf16 intermediate (faster but lossy)

| N | fp16 direct | fp16→bf16 | speedup | relative error |
|---|---|---|---|---|
| 1024 | 6.780 ms | **1.611 ms** | 4.21× | **0.38%** |
| 2048 | 54.273 ms | **14.297 ms** | 3.80× | **0.37%** |

bf16 has only 8 mantissa bits, hence the 0.38% error. **Unless that precision is acceptable, use the fp32 intermediate.**

---

## 5. Option two: blocked GEMM to bypass the 4096 cliff (this report's biggest gain)

Principle: for N≥3072 cuBLAS's heuristic picks a bad kernel; split the large GEMM into small blocks that land in the
sweet spot.

```python
def mx450_blocked_mm(a, b, bs=512):
    n = a.shape[0]
    c = torch.empty(n, n, dtype=a.dtype, device=a.device)
    for i in range(0, n, bs):
        ai = a[i:i+bs, :]
        for j in range(0, n, bs):
            bj = b[:, j:j+bs].contiguous()
            c[i:i+bs, j:j+bs] = torch.mm(ai, bj)
    return c
```

### fp32 blocking measurements

| N | block | Time | TFLOPS | vs whole |
|---|---|---|---|---|
| 2048 | whole | 8.305 ms | 2.0687 | 1.00× |
| 2048 | 512 | 13.357 ms | 1.2862 | 0.62× |
| 2048 | 1024 | 9.110 ms | 1.8857 | 0.91× |
| 2048 | 2048 | 8.682 ms | 1.9787 | 0.96× |
| 3072 | whole | 62.274 ms | 0.9311 | 1.00× |
| 3072 | 512 | 45.000 ms | 1.2885 | **1.38×** |
| 3072 | **1024** | **42.149 ms** | **1.3757** | **1.48×** |
| 4096 | whole | 369.192 ms | 0.3723 | 1.00× |
| 4096 | **512** | **82.459 ms** | **1.6668** | **4.48×** |
| 4096 | 1024 | 95.256 ms | 1.4428 | 3.88× |
| 4096 | 2048 | 134.701 ms | 1.0203 | 2.74× |

**N=4096, block=512 → 369 ms drops to 82 ms, 4.48×.**

Blocked `err` is on the order of 1e-3, the normal floating-point error from a changed fp32 accumulation order
(values around 1000, relative error ~1e-6), not a bug.

### fp16 blocking: ineffective, as expected

| N | whole | block=1024 | Effect |
|---|---|---|---|
| 2048 | 54.272 ms | 54.925 ms | 0.99× |
| 4096 | 400.736 ms | 438.842 ms | 0.91× |

**Proving that fp16's bottleneck is HMMA emulation, not the cuBLAS heuristic** — blocking cannot save it; only
changing dtype can.

---

## 6. Four rules that can be applied directly

| # | Rule | Basis | Gain |
|---|---|---|---|
| **1** | **never use fp16 direct in GEMM** | cutlass_70_tensorop_f16_s884 + the PTX documentation warning | avoids a 7.5× loss |
| **2** | **fp16 requirement → convert to fp32 → compute → convert back** | measured 3.3–3.7× at 1024/2048, err=0 | 3.3–3.7× |
| **3** | **use blocked GEMM (bs=512) when N≥3072** | measured 4.48× at 4096 | 1.4–4.5× |
| **4** | **bf16 on Turing is fp32 emulation, but with only 8 mantissa bits** | magma_sgemmEx + rel=0.38% | 4.2× but lossy |

### Combined strategy (recommended)

```python
def mx450_gemm(a, b):
    """The optimal GEMM path on an MX450 (fp16 in → fp32 compute → fp16 out)"""
    n = a.shape[-1]
    fp32 = a.dtype in (torch.float16, torch.bfloat16)
    x, y = (a.float(), b.float()) if fp32 else (a, b)
    bs = 512 if x.shape[0] >= 3072 else 0
    c = mx450_blocked_mm(x, y, bs) if bs else torch.mm(x, y)
    return c.half() if fp32 else c
```

### Appendix: framework-level settings

```python
# on Turing, autocast's bf16/fp16 GEMM falls into the bad path; use fp32 for training
torch.backends.cuda.matmul.allow_tf32 = False   # no TC, so TF32 is meaningless
# measured: allow_fp16_reduced_precision_reduction and allow_tf32 have no effect on fp16 mm (1.00×)
```

---

## 7. Relationship to the other results

| Result | TFLOPS | Relative to cuBLAS fp16 |
|---|---|---|
| cuBLAS fp16 (bad path) | 0.32 | 1.0× |
| cuBLAS fp32 (sweet spot) | 2.37 | 7.5× |
| **self-built software Tensor Core** | **1.603** | **5.0×** |
| theoretical physical peak (FP32 FMA) | 3.06 | 9.6× |

**Conclusion**: the self-built software TC (1.603) is still significantly better than cuBLAS fp16 (0.32),
but **worse than cuBLAS fp32 in its sweet spot (2.37)**.
For N=1024/2048 scenarios, **using cuBLAS fp32 directly is the current optimum**;
the custom kernel's value lies in cases where the input and output are fp16 and fp16 precision must be preserved.

---

## 8. Reproduction scripts

- `full-unlock/04-results/cublas_path_study.py` — size sweep + mixed paths + kernel names + switches
- `full-unlock/04-results/blocked_gemm_study.py` — blocked GEMM + bf16 intermediate

Run:
```bash
python nas_exec.py NVDIA-MX450-pojie/full-unlock/04-results/blocked_gemm_study.py
```
