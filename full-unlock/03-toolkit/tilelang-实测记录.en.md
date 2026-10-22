# TileLang — a real trial record

> The user demanded "you must actually run it". This document is the complete measurement of TileLang on the MX450,
> including three failures and a final success.

---

## 1. Why it was tried

**The search results stated plainly:**

> **"tile-lang supports CUDA devices with architectures including Ampere (sm_80+),
> Turing (sm_75), and Volta (sm_70)."**

**This is the only high-performance GEMM DSL that explicitly claims sm_75 support.
If it runs, it is the strongest reference point against our hand-written kernel.**

---

## 2. Four attempts, three failures

### Failure 1 · target detection failed

```
pip install tilelang          # succeeded, 0.1.15
```

```python
kernel = tilelang.compile(func, out_idx=[2])
```

```
FAIL: No registered target detector found an available target.
```

**Cause: TVM did not auto-detect the CUDA backend.**

**Fix: specify `target="cuda"` explicitly.**

### Failure 2 · missing CUDA_HOME

```
FAIL: Failed to automatically detect CUDA installation.
      Please set the CUDA_HOME environment variable manually.
```

**On inspection: pip's `nvidia-cuda-nvcc-cu12` package ships only `ptxas`, not `nvcc`.**

```
nvidia/cuda_nvcc/bin/ptxas      ← only this
nvidia/cuda_nvcc/include/...    ← only headers
```

**This is an inherent limitation of the pip package: it does not provide the complete nvcc compiler.**

**Fix: install the standalone component from NVIDIA's official apt repository (not the whole toolkit).**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-nvcc-12-9      # compiler only, about 27 MB
```

**Result: `/usr/local/cuda-12.9/bin/nvcc` (27 MB), version 12.9.86.**

**Note: `cuda-nvcc-12-1` does not exist; the available ones are 12-5 / 12-6 / 12-8 / 12-9 / 13-0 to 13-3.**

### Failure 3 · DLPack allocator

```
FAIL: TVM-FFI callee-allocated outputs require Torch's DLPack allocator exchange API.
      Install a compatible torch-c-dlpack-ext or use a supported PyTorch build.
```

**Cause: `out_idx=[2]` makes TileLang allocate the output tensor itself, which requires `torch-c-dlpack-ext`.**

**Fix (faster than reinstalling packages): switch to a caller-allocated output.**

```python
# before
kernel = tilelang.compile(func, out_idx=[2], target="cuda")
c = kernel(a, b)

# after
kernel = tilelang.compile(func, target="cuda")
c = torch.empty(M, N, device="cuda", dtype=torch.float16)
kernel(a, b, c)
```

### Success · correctness + performance

```
correctness (512³):
  max abs err = 0.06250  (relative 5.56e-04)   PASS

performance (2048³, float16, forced synchronisation):
  (128,128,32)  compile 1.5s   0.527 TFLOPS  (32.6 ms)
  (64,128,32)   compile 1.4s   0.538 TFLOPS  (31.9 ms)   ← best
  (128,128,64)  FAIL  shared memory 98304 > limit
  (128,256,32)  FAIL  shared memory 73728 > limit
```

---

## 3. A hardware constraint got exposed

**Two large-tile configurations failed because shared memory exceeded the limit:**

```
Failed to set the allowed dynamic shared memory size to 98304   (96 KB)
Failed to set the allowed dynamic shared memory size to 73728   (72 KB)
```

**The MX450 (Turing) SM shared-memory limit is 64 KB (96 KB requires opt-in).**

**TileLang does not perform that opt-in for sm_75, so configurations like block_K=64 fail outright.**

**This means: on this card, the usable tile configurations for TileLang are significantly curtailed.**

---

## 4. Complete side-by-side comparison (all forced synchronisation)

| Implementation | TFLOPS | vs hand-written | Path taken |
|---|---|---|---|
| cuBLAS `TENSOR_OP` | 0.320 | 0.20× | HMMA (emulated) |
| tinygrad (after sync) | 0.347 | 0.22× | HMMA (inferred) |
| **TileLang (best)** | **0.538** | **0.34×** | **HMMA (cute/CUTLASS)** |
| **hand-written CUDA C here (FFMA)** | **1.603** | **1.00×** | **FFMA** |
| cuBLAS FP32 | 2.122 | 1.32× | FFMA |

**Key observation:**

**Every implementation going through HMMA (cuBLAS TENSOR_OP / tinygrad / TileLang) lands in the 0.32–0.54 band.**

**Every implementation going through FFMA (hand-written / cuBLAS FP32) lands in the 1.60–2.12 band.**

**That is not a coincidence; it is the direct consequence of TU117 having no real TCs.**

---

## 5. Conclusion

### TileLang works, but it is not suited to this card

| Aspect | Assessment |
|---|---|
| sm_75 support | ✅ officially claimed, and it compiles and runs in practice |
| correctness | ✅ PASS (relative error 5.56e-04, reasonable for FP16) |
| performance | ❌ 0.538, i.e. 34% of hand-written |
| large tiles | ❌ 64 KB shared-memory limit, several configurations fail |
| installation barrier | ⚠️ needs a full nvcc (the pip package is not enough) |

**Root cause: `T.gemm` dispatches to cute/CUTLASS on NVIDIA, and CUTLASS defaults to HMMA on sm_75.
TU117's HMMA is emulated, hence the slowness.**

**If TileLang offered a "disable tensor core" option (similar to `tl.dot`'s input_precision),
it could in theory take the FFMA path. The current version does not expose that switch.**

### These three failures plus one success jointly prove one thing

**On TU117, "using a high-level DSL" is not the optimal solution.
The only way to reach 1.6 TFLOPS is to write CUDA C yourself and force the FFMA path.**

**This project's `sw_tensor_core_v2.py` is the product of that conclusion.**

---

## 6. Environment record

```
tilelang    0.1.15
torch       2.3.1+cu121
nvcc        12.9.86 (installed separately from NVIDIA's apt repository)
CUDA_HOME   /usr/local/cuda-12.9
GPU         MX450, 14 SM, CC 7.5
shared-mem limit  64 KB (96 KB requires opt-in)
```

**Reproduce:**
```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
python test_tilelang.py
```

**File: `full-unlock/03-toolkit/test_tilelang.py`**
