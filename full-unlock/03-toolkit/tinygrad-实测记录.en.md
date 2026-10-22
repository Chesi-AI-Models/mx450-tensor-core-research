# tinygrad — a real trial record (including a major measurement trap)

> The user demanded "actually run it, don't judge from the documentation". This is the actual execution record for tinygrad.
>
> **Core finding: tinygrad's lazy evaluation inflates performance measurements by 23×.**

---

## 1. Install and first test

```bash
pip install tinygrad      # succeeded, via PyPI (direct GitHub access is unstable on the NAS)
```

```python
from tinygrad import Tensor, Device
print(Device.DEFAULT)     # CUDA
```

**Backend detection correct (CUDA).**

---

## 2. First performance measurement: 8.0 TFLOPS — but it is fake

### Measurement code

```python
N = 2048
a = Tensor.rand(N, N, dtype="float32")
b = Tensor.rand(N, N, dtype="float32")
(a @ b).realize()          # first compile

t0 = time.time()
for _ in range(10):
    c = (a @ b).realize()  # ← the problem is here
dt = (time.time() - t0) / 10
```

### Results

```
float32:  7.638 TFLOPS   (2.25 ms)
float16:  4.686 TFLOPS   (3.67 ms)
```

### Why it was suspect

| Check | Data | Judgement |
|---|---|---|
| MX450 FP32 theoretical peak | 896 CUDA × 2 FMA × 1.6 GHz = **2.87 TFLOPS** | 7.64 is **2.7×** that |
| Frequency required | 7.638e12 / (896×2) = **4.26 GHz** | impossible for an MX450 |
| cuBLAS FP32 measured | **2.122 TFLOPS** | tinygrad cannot be 3.6× faster |
| PTX FP32 FMA pure-compute peak | **3.06 TFLOPS** | 7.64 exceeds the pure-compute ceiling |

**Four independent reference points all contradict it → it must be a measurement problem.**

---

## 3. Locating the problem: tinygrad is lazy

**tinygrad's design centre is lazy evaluation**: operators are first recorded into a UOp graph and only actually
executed when a result is needed.

**`.realize()` merely "turns the graph into kernels and *queues* them"; it does not guarantee execution has finished.**

**So the timing measures "queueing + launch overhead", not "kernel execution time".**

**The larger the matrix, the bigger the inflation — because the real computation is pushed outside the timing window.**

---

## 4. Real numbers after forcing synchronisation

```python
# method 1: realize only (async)
t0 = time.time()
for _ in range(5):
    c = (a @ b).realize()
t_realize = (time.time() - t0) / 5

# method 2: realize + numpy (forces a copy back, hence synchronisation)
t0 = time.time()
for _ in range(5):
    c = (a @ b).numpy()
t_numpy = (time.time() - t0) / 5

# method 3: realize + Device.synchronize()
dev = Device[Device.DEFAULT]
t0 = time.time()
for _ in range(5):
    c = (a @ b).realize()
    dev.synchronize()
t_sync = (time.time() - t0) / 5
```

### Results (2048³, float32)

| Method | Time | TFLOPS | Verdict |
|---|---|---|---|
| `realize()` async | 2.14 ms | **8.029** | ❌ fake |
| `realize()+numpy()` | 116.64 ms | **0.147** | ✅ real (includes D2H copy) |
| `realize()+synchronize()` | 49.44 ms | **0.347** | ✅ real (pure GPU time) |

**Real performance: 0.347 TFLOPS. Inflated 23×.**

---

## 5. Corrected full comparison

| Implementation | TFLOPS | Relative to cuBLAS TENSOR_OP |
|---|---|---|
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | 0.320 | 1.0× |
| **tinygrad (after sync)** | **0.347** | **1.08×** |
| **hand-written CUDA C here (FFMA, NVRTC)** | **1.603** | **5.0×** |
| cuBLAS FP32 | 2.122 | 6.6× |

**Conclusion: tinygrad's matmul is on par with cuBLAS TENSOR_OP (0.32–0.35),
showing it likewise takes the HMMA path (emulated TC on TU117).**

**And the hand-written FFMA kernel is 4.6× faster.**

**This in turn validates the project's core result: on a card without TCs, bypassing HMMA and going FFMA is the
correct strategy.**

---

## 6. Correctness verification

```python
N = 512
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
ref = A @ B

c = (Tensor(A) @ Tensor(B)).numpy()
err = np.abs(c - ref).max()      # 0.00007
rel = err / np.abs(ref).max()    # 6.91e-07
```

**Verdict: PASS (relative error 6.9e-07, as expected for FP32).**

**Note: tinygrad's computation is correct; only the timing method is a trap.**

---

## 7. How widespread this trap is

**Any performance comparison made with tinygrad must confirm whether synchronisation was added.**

**Every tinygrad tutorial/benchmark found in the search only demonstrates `DEBUG=4` for inspecting kernels and never
mentions this.**

**This is not a tinygrad bug (laziness is its design), but a pitfall users easily fall into.**

**Recorded here for future reference.**

---

## 8. Practical conclusions for this project

**What tinygrad is worth on an MX450:**

| Aspect | Assessment |
|---|---|
| performance | ❌ worse than hand-written FFMA (0.347 vs 1.603) |
| correctness | ✅ passes |
| debuggability | ✅ excellent (`DEBUG=4` shows the generated kernels directly) |
| TU117 fit | ⚠️ takes HMMA, no optimisation for the absence of TCs |
| value as a reference implementation | ✅ worth reading its CUDA backend code |

**Conclusion: tinygrad cannot be used directly to accelerate the MX450, but its `DEBUG=4` lets you inspect kernel
generation, which is useful reference material for hand-written optimisation.**

---

## 9. Reproduce

```bash
pip install tinygrad
python verify_tinygrad.py           # correctness + performance (reproduces the async trap)
python verify_tinygrad_sync.py      # comparison of the three synchronisation methods
```

**File locations:**
- `full-unlock/03-toolkit/test_tinygrad.py` (first test)
- `full-unlock/03-toolkit/verify_tinygrad.py` (correctness verification)
- `full-unlock/03-toolkit/verify_tinygrad_sync.py` (synchronisation comparison)
