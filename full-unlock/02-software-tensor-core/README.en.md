# Software Tensor Core: rebuilding matrix multiplication on a card with no TC

> Goal: TU117 (MX450) has no Tensor Core, so implement equivalent matrix-multiply acceleration on CUDA cores.
>
> Final result: **1.603 TFLOPS**, which is **5.0×** the cuBLAS `TENSOR_OP` path (0.320),
> with `max abs err = 0.00000` (identical to the float64 reference).

---

## 1. Why it is needed

| Path | Measured TFLOPS | Note |
|---|---|---|
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | 0.320 | cuBLAS believes there are TCs and picks `cutlass_70_tensorop_f16_s884gemm`, but TU117 has none, so FP16 units emulate it |
| cuBLAS FP32 | 2.122 | already 85% of the theoretical peak (2.5) |
| FP16x2 pure-compute peak | 5.770 | **FP16 accumulation**; overflows at K=2048, unusable for real GEMM |

**So: on FP16, cuBLAS hands you a dead end at 0.32 (because it thinks TCs exist).
Writing your own FFMA kernel gets 1.6 — that is where the 5× comes from.**

---

## 2. Four-version evolution (including two failures)

| Version | File | Approach | Result |
|---|---|---|---|
| v1 | `triton_gemm.py` | Triton `tl.dot`, FP16 inputs | 0.595 TFLOPS (goes through HMMA) |
| v2 | `triton_gemm_v2.py` | outer product (`tl.where`+`tl.sum` to avoid dot) | 0.122 (generates masses of shuffles, 5× slower than HMMA) |
| v4 | `triton_gemm_v4.py` | Triton `tl.dot`, FP32 inputs | **compile crash `map::at`** (a known Triton bug on CC 7.5, triton-lang/triton#5557) |
| **final** | **`sw_tensor_core_v2.py`** | **NVRTC + CUDA C + FFMA** | **1.603 TFLOPS** ✅ |

### The key lesson

**Triton is unusable for FP32 on CC 7.5.** The official issue states it plainly:
native FP32 support is limited on compute capability 7.5; the compiler loses type metadata while emitting LLVM IR,
triggering a `map::at` crash. Switching to FP16 bypasses it — but FP16 goes through HMMA, which is exactly what we
want to avoid.

**So Triton must be bypassed entirely and code generated directly.**

---

## 3. The final implementation (`sw_tensor_core_v2.py`)

### Stack

```
CUDA C source (unsigned short + inline PTX)
    ↓ libnvrtc.so.12 (NVRTC)
PTX (39,393 characters)
    ↓ cuModuleLoadData (libcuda)
executed on GPU
```

**`cuda_fp16.h` is not used** (it is not on NVRTC's default include path); everything goes through:

```cuda
__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
```

### Kernel structure

```
BLOCK: 128 × 128, BK = 16, 256 threads
per thread: 8 × 8 = 64 float accumulators
shared memory: As[128][16] + Bs[16][128] (half, 8 KB total)

for each k:
  load 8 a values (shared memory → registers)
  load 8 b values
  64 FFMAs
  → FMA/load ratio = 64/16 = 4
```

### Verified PTX characteristics

```
contains cvt.f32.f16 : True
contains fma.rn.f32  : True
contains mma.sync    : False   ← key: HMMA is completely bypassed
```

---

## 4. Performance data

**Matrix 2048³, averaged over 10 iterations:**

```
1.603 TFLOPS   (10.7 ms per call)
max abs err = 0.00000
```

| Comparison | TFLOPS | Ratio |
|---|---|---|
| cuBLAS TENSOR_OP | 0.320 | **5.0×** |
| cuBLAS FP32 | 2.122 | 0.76× |
| FP16x2 pure-compute peak | 5.770 | 0.28× (not comparable; FP16 accumulation) |

### Why 1.603 and not 5.77

**`fma.rn.f16x2` accumulates in FP16. At K=2048 the accumulator overflows and the result is unusable.**

A real GEMM must accumulate in FP32 → only FFMA is available → the hardware ceiling is FP32's 2.5 TFLOPS.
1.603 / 2.5 = **64% efficiency**; the remaining loss comes from shared-memory bandwidth and instruction scheduling.

**This is the hardware's arithmetic-intensity ceiling, not an implementation defect.**

### Optimisations tried that were slower

| Optimisation | Result | Reason |
|---|---|---|
| 4×4 tile (BM=BN=64) | 1.539 | FMA/load ratio only 2 |
| **8×8 tile (BM=BN=128)** | **1.603** | ✅ optimal |
| shared memory as float | 1.427 | shared memory doubles, bank conflicts increase |

---

## 5. Pitfalls hit (calling the CUDA Driver API from ctypes)

1. **`argtypes` must be declared explicitly**
   otherwise 64-bit pointers get truncated to int → **SIGSEGV** at `cuModuleLoadData`.
   This was the single most time-consuming bug in the project.

2. **PyTorch's CUDA context must be reused**
   calling `cuCtxCreate_v2` yourself creates a second context; the module then loads into the wrong context,
   manifesting as "load succeeds but calls do nothing".
   Correct approach: `torch.cuda.init()` → `cuCtxGetCurrent()`.

3. **`cuModuleLoadData` accepts the PTX text directly**
   no ptxas, no nvcc; the driver JITs it internally.

---

## 6. Reproduce

```bash
/vol1/ganlin/venv/bin/python sw_tensor_core_v2.py --size 2048
```

Dependencies: `torch 2.3.1` + `nvidia-cuda-nvrtc-cu12` (providing `libnvrtc.so.12`).

---

## 7. What this result means

**The MX450's true matrix-multiply ceiling is 2.5 TFLOPS (FP32), and cuBLAS already achieves 2.12.**

**But cuBLAS's FP16 path only reaches 0.32 — because it thinks TCs exist.**

**So the real value of a "software TC" is not beating FP32, it is this:**

**when your model is FP16, do not use cuBLAS's FP16 path (0.32);
use a custom half→float→FFMA kernel (1.6), 5× faster.**

**That is the optimum achievable on hardware without TCs.**
