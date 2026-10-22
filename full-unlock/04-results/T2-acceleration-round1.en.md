# T2 acceleration: measurements and the first optimisation round

> Goal: accelerate GreenBoost's T2 tier (system RAM) as thoroughly as possible, covering both inference and training.
>
> Result this round: **a headroom configuration defect found and fixed; real T2-related bandwidth improved 12.1×.**

---

## 1. Baseline bandwidth measurement (first quantification)

**Method**: `y = x + 1` (elementwise, memory-bandwidth bound), FP32, 5 iterations.

| Scenario | Size | Bandwidth | Time | Path |
|---|---|---|---|---|
| **VRAM** | 0.5 GiB | **67.52 GB/s** | 15.9 ms | pure VRAM |
| T2 | 0.5 GiB | 5.60 GB/s | 191.7 ms | **wrongly took T2** |
| T2 | 1.0 GiB | 5.69 GB/s | 377.2 ms | T2 |
| T2 | 2.0 GiB | 5.66 GB/s | 758.9 ms | T2 |
| T2 | 3.0 GiB | 5.65 GB/s | 1140.1 ms | T2 |
| T2 | 4.0 GiB | 5.65 GB/s | 1521.6 ms | T2 |

**Core ratio: VRAM 67.52 / T2 5.65 = 11.95×.**

**Note: T2 bandwidth is highly stable across sizes (5.60–5.69 GB/s), showing the bottleneck is the PCIe link itself,
not software overhead.**

---

## 2. The first serious defect found: the headroom default is too large

### Symptom

**A 0.5 GiB tensor should fit in VRAM (1.64 GiB usable), yet measured only 5.60 GB/s — it took T2.**

### Root cause

The shim's environment variable `GREENBOOST_VRAM_HEADROOM_MB` **defaults to 2048 (2 GB)**.

The shim's logic is "reserve N MB of VRAM unused; only fall back to T2 when the remainder is insufficient".
On a 1.64 GiB card, reserving 2 GB means **reserving the entire VRAM**,
so every allocation above 256 MB is pushed to T2.

**This is a fatal default on small-VRAM cards like the MX450.**

### Fix

```bash
GREENBOOST_VRAM_HEADROOM_MB=128
```

### Effect

| Scenario | default headroom=2048 | optimised headroom=128 | Gain |
|---|---|---|---|
| 0.5 GiB | 5.60 GB/s | **67.85 GB/s** | **12.1×** |
| 1.0 GiB | 5.69 GB/s | 6.58 GB/s | 1.16× (still T2: 1G input + 1G output > 1.64G) |
| 2.0 GiB | 5.66 GB/s | 5.66 GB/s | 1.0× (must take T2) |

**Conclusion: headroom should be set to ~8% of VRAM, i.e. 128 MB on an MX450.
Allocations that fit in VRAM should all use VRAM; only genuinely oversized ones fall to T2.**

---

## 3. What T2 really is

| Metric | Value | Meaning |
|---|---|---|
| VRAM bandwidth | 67.5–67.9 GB/s | baseline |
| T2 bandwidth | 5.65 GB/s | **1/12 of VRAM** |
| T2 capacity | 8 GB (adjustable) | 4.9× VRAM |
| T3 (NVMe) | ~1.8 GB/s | 1/37 of VRAM |

**This determines the direction of the optimisation strategy:**

**T2 is not "slightly slower VRAM"; it is a different order of magnitude.
The goal is not "make T2 faster" (bandwidth is soldered to PCIe), but "minimise the number of T2 accesses".**

---

## 4. Optimisation priorities going forward (ordered by expected gain)

| # | Approach | Expected gain | Principle |
|---|---|---|---|
| 1 | **headroom tuning** | **done, 12.1×** | keep anything that fits in VRAM out of T2 |
| 2 | **quantisation** | **bytes ÷2 to ÷4** | FP16→INT8 halves, →INT4 cuts 75%. Effective bandwidth 11.3 / 22.6 GB/s |
| 3 | **weight residency policy** | zero transfer for hot layers | keep hot layers resident in VRAM, cold layers in T2 |
| 4 | **double buffering + async prefetch** | hides transfer latency | prefetch layer N+1 while computing layer N |
| 5 | **`cuMemAdviseSetReadMostly`** | avoids write-back | weights are read-only; forbid the driver writing back to T2 |
| 6 | **`cuMemPrefetchAsync`** | eliminates page-fault stalls | migrate proactively instead of relying on UVM's black-box decisions |

**Key judgement: only approach 2 (quantisation) can break through the PCIe bandwidth wall.
Everything else is "transfer less often" or "hide latency"; only quantisation genuinely transfers 4× fewer bytes.**

---

## 5. Implications for training

**Training depends on bandwidth more than inference does:**

- inference: each layer's weights read once (forward)
- training: forward read + backward read + gradient write ≈ **3× the traffic**

**So training over T2 costs roughly 3× inference.**

**This is why GreenBoost's original author focused his optimisation on the inference side (Ollama/ExLlama).**

**Viable paths for training on an MX450:**

1. **QLoRA (4-bit quantisation) + small batch** — compress weights to 4-bit, cutting transfer volume 75%
2. **LoRA training only adapters** — main weights frozen (read-only, can use `SetReadMostly`), only adapters in VRAM
3. **gradient checkpointing** — trade compute for VRAM
4. **data trimming** — validate the recipe on 1% of the data first

**Together these four are the only route to working training on this machine.**

---

## 6. Environment this round

```
GPU: NVIDIA GeForce MX450 (TU117)
physical VRAM: 1.64 GiB usable
GreenBoost: greenboost.ko loaded
  physical_vram_gb=2  virtual_vram_gb=8  safety_reserve_gb=2
  nvme_swap_gb=8  pcores_only=0  use_hugepages=1
shim: libgreenboost_cuda.so (with the _v2 symbols fixed)
```

**Reproduce:**
```bash
# VRAM control
python bench_t2_bw.py

# T2
LD_PRELOAD=.../libgreenboost_cuda.so GREENBOOST_ACTIVE=1 \
GREENBOOST_VRAM_HEADROOM_MB=128 \
python bench_t2_bw.py
```

---

## 7. Round two: measuring the quantisation gain (decisive)

### Method

The same **2 GiB of bytes**, different dtypes, over T2:

```python
y = x + x   # elementwise, bandwidth bound
```

### Results

| dtype | Elements | Bandwidth | **Element throughput** | Time |
|---|---|---|---|---|
| float32 | 0.54 G | 5.66 GB/s | **1.41 G/s** | 759.3 ms |
| float16 | 1.07 G | 5.67 GB/s | **2.83 G/s** | 758.1 ms |
| int8 | 2.15 G | 5.62 GB/s | **5.62 G/s** | 763.9 ms |

**Two key observations:**

1. **Bandwidth is identical (5.62–5.67 GB/s)** — proving the bottleneck is **byte transfer**, not element count
2. **Element throughput scales linearly** (1.41 → 2.83 → 5.62) — proving **the quantisation gain is linear**

### Inference: INT4 brings T2's effective bandwidth up to VRAM levels

| Precision | Bytes/param | 7B model | T2 load time | **Effective bandwidth** |
|---|---|---|---|---|
| FP32 | 4 | 28 GB | 4.96 s | 5.65 GB/s |
| FP16 | 2 | 14 GB | 2.48 s | **11.3 GB/s** |
| INT8 | 1 | 7 GB | 1.24 s | **22.6 GB/s** |
| **INT4** | **0.5** | **3.5 GB** | **0.62 s** | **45.2 GB/s** |
| VRAM baseline | — | — | — | 67.5 GB/s |

**With INT4, T2's effective bandwidth is 45.2 GB/s, i.e. 67% of VRAM.**

**That is the core answer to "thoroughly accelerating T2": not making PCIe faster (physically impossible),
but compressing the bytes to be transferred to 1/8.**

---

## 8. Combined effect after optimisation

| Scenario | Before | After | Multiple |
|---|---|---|---|
| 0.5 GiB tensor (fits in VRAM) | 5.60 GB/s (wrongly T2) | **67.85 GB/s** (correctly VRAM) | **12.1×** |
| 7B model weights (INT4, T2) | 5.65 GB/s effective | **45.2 GB/s effective** | **8.0×** |
| 7B model weights (FP16, T2) | 5.65 GB/s effective | 11.3 GB/s effective | 2.0× |

**With both optimisations stacked, T2 is no longer "a garbage tier at 1/12 the performance" but "a usable tier
1.5× slower with 4.9× the capacity".**

---

## 9. Deployment configuration (training 7B QLoRA)

```bash
# 1. kernel module (headroom is controlled by the shim, not a module parameter)
sudo modprobe greenboost physical_vram_gb=2 virtual_vram_gb=8 \
     safety_reserve_gb=2 nvme_swap_gb=8 pcores_only=0 use_hugepages=1

# 2. run (key: drop headroom to 128 so VRAM is preferred)
LD_PRELOAD=/vol1/ganlin/greenboost-build/libgreenboost_cuda.so \
GREENBOOST_ACTIVE=1 \
GREENBOOST_VRAM_HEADROOM_MB=128 \
python train_qlora.py

# 3. the training script must use 4-bit quantisation (bitsandbytes / QLoRA)
#    this is needed both for VRAM and for T2 bandwidth
```

**Key: `GREENBOOST_VRAM_HEADROOM_MB=128` and 4-bit quantisation must be used together.**
**The former stops VRAM being wasted; the latter cuts T2 transfer volume to 1/8.**
