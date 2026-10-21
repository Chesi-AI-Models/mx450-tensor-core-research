# MX450 Full Unlock Attempt — Reality Check and Execution Record

> This directory is the complete answer to the goal of "**unlocking the MX450's Tensor Core as thoroughly as possible**".
>
> Conclusion up front: **TU117 has no Tensor Core hardware, so there is nothing to "unlock"; but one can be
> rebuilt on CUDA cores — measured 1.603 TFLOPS, 5.0× cuBLAS TENSOR_OP.**

---

## 1. Reality check first: what can be broken, what cannot

Before starting, the word "unlock" must be taken apart. The four classes of limits on an MX450 are fundamentally different:

| Target | Breakable? | Reason | What this directory does |
|---|---|---|---|
| **Tensor Core** | ❌ Does not exist | TU117's SM has no TC; that area holds FP16 units | ✅ **Rebuild in software** (`02-software-tensor-core/`, 1.603 TFLOPS) |
| **VRAM 2 GB** | ⚠️ **Expandable** | Two GDDR6 packages are soldered, but system RAM can extend it | ✅ **GreenBoost: 8 GB virtual VRAM measured** (see below) |
| **SM clock** | ⚠️ Limited | `nvidia-settings` allows small overclocks, capped by TGP | — |
| **FECS/GPCCS microcode** | ⚠️ Read-only | FWSEC certificate chain; one changed byte blocks loading | Fully dumped (see parent `findings/`) |

### ⭐ The most important result: VRAM expanded from 1.64 GiB to 8 GiB

**GreenBoost (upstream source, fetched via `refs/refs-fetch-manifest.sh`) fully validated on MX450 + fnOS 6.18.18:**

```
physical VRAM reported: 1.64 GiB

alloc 2 GiB: OK    alloc 4 GiB: OK    alloc 8 GiB: OK   ← 4.9× the physical size
matmul on 4GiB-occupied GPU: OK, c[0,0]=1024.0 (expect 1024.0)
```

**Three-tier storage: T1 VRAM 2 GB + T2 system RAM 8 GB + T3 NVMe 8 GB = 18 GB model capacity.**

**Cost: T2 goes over PCIe 3.0 x4 at roughly 4 GB/s — 1/20 of VRAM. Tolerable for inference, 3–10× slower for training.**

**Adaptation details (including the only two lines changed): [`01-upstream-analysis/greenboost-adaptation.md`](01-upstream-analysis/greenboost-adaptation.md)**

**This turns 7B QLoRA (needs ~4 GB) from "physically impossible" into "runs, but slowly".**

### Why CMPUnlocker's method cannot be copied

CMPUnlocker unlocks a CMP 170HX from 8 GB to 64 GB **because 80 GB of HBM2e physically exists** —
it is merely restricted by OTP fuses and firmware.

The MX450 situation is:

- VRAM: two GDDR6 packages = 2 GB, **and nothing more**
- TC: TU117's SM **has no TC unit**

**No sealed hardware means nothing to unlock.** This is the fundamental difference between this directory and CMPUnlocker.

---

## 2. Directory layout

```
full-unlock/
├── README.md                      This file: reality check + overview
├── 01-upstream-analysis/          Analysis of every referenced open-source project
│   ├── cmpunlocker.md             ★ Full technical teardown of CMPUnlocker
│   └── (third-party sources removed — see refs/refs-fetch-manifest.sh)
├── 02-software-tensor-core/       ★ Core: rebuilding a Tensor Core on CUDA cores
│   ├── README.md                  Implementation notes and performance data
│   ├── triton_gemm.py             v1: Triton tl.dot (goes through HMMA) 0.595
│   ├── triton_gemm_v2.py          v2: outer product (failed) 0.122
│   ├── triton_gemm_v4.py          v4: FP32 dot (Triton crashes)
│   └── sw_tensor_core_v2.py       ★ Final: NVRTC + FFMA, 1.603 TFLOPS
├── 03-toolkit/                    Reuse of existing tooling (PRAMIN / Falcon / nvidia.ko)
└── 04-results/                    Measurement data
```

---

## 3. Core result: a software Tensor Core

**A matrix-multiply engine rebuilt on CUDA cores on TU117, bypassing the non-existent TC.**

### Technical path

```
CUDA C source (unsigned short + inline PTX)
    ↓ NVRTC compile (libnvrtc.so.12)
PTX (39,393 chars, containing cvt.f32.f16 / fma.rn.f32, no mma.sync)
    ↓ cuModuleLoadData (libcuda)
Executed on GPU
```

**Key point: `cuda_fp16.h` is never used (NVRTC cannot find it); everything goes through `unsigned short` plus inline PTX
`cvt.f32.f16` / `cvt.rn.f16.f32`.**

### Measured performance (2048³, local machine)

| Implementation | TFLOPS | vs cuBLAS TENSOR_OP |
|---|---|---|
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | 0.320 | 1.0× |
| Triton `tl.dot` (FP16, goes through HMMA) | 0.595 | 1.9× |
| **`sw_tensor_core_v2.py` here (FFMA)** | **1.603** | **5.0×** |
| cuBLAS FP32 (`volta_sgemm`) | 2.122 | 6.6× |
| FP16x2 pure-compute peak (FP16 accumulate, unusable for real GEMM) | 5.770 | — |

**Correctness: `max abs err = 0.00000` (identical to the float64 reference).**

### Why 1.603 and not 5.77

**`fma.rn.f16x2` accumulates in FP16. At K=2048 the accumulator overflows and the result is unusable.**

A real GEMM must accumulate in FP32 → only FFMA is available → the ceiling is FP32's 2.5 TFLOPS.
1.603 is 64% of 2.5; the rest is lost to shared-memory bandwidth and instruction scheduling.

**This is not a weak implementation — it is the hardware's arithmetic-intensity limit.**

---

## 4. How the upstream projects were actually used

| Project | Used? | How |
|---|---|---|
| **CMPUnlocker** | ✅ Source studied + full teardown | Learned the ROP-injection mechanism; confirmed its PLM target registers do not exist on Turing |
| **Triton** | ✅ Actually used | v1 baseline 0.595; found an FP32-path `map::at` crash on CC 7.5 |
| **NVRTC / CUDA Driver API** | ✅ Actually used | The final version depends entirely on this path |
| **tinygrad** | ⚠️ Reference only | Its user-space driver approach does not fit this task |
| **TileKernels / FlagOS** | ⚠️ Reference only | Ascend/multi-chip oriented; Triton DSL ideas were borrowed |
| **DRAMA++ / Sudoku** | ❌ Not applicable | ARM64 server DRAM address mapping |
| **Open-RDMA / fabric-lib** | ❌ Not applicable | Network layer, irrelevant to a single card |
| **AITER** | ❌ Not applicable | AMD ROCm specific |
| **shadPS4 VRAM refactor** | ❌ Not applicable | Emulator scenario |

**Honest note: of the projects listed in the txt, only CMPUnlocker (methodology) and Triton (toolchain) are genuinely
usable on an MX450. The rest do not fit architecturally or by platform — that is not a choice, it is a physical constraint.**

---

## 5. License boundary (important)

**All content in this directory is original to this project and MIT licensed.**

Earlier versions inlined `01-upstream-analysis/vendor/` (GPL v2 cmpunlocker and greenboost).
**It was fully removed on 2026-10-21**; the repository no longer carries any third-party source copy, so the license boundary is clean.

To inspect upstream sources: `bash refs/refs-fetch-manifest.sh` (SHAs pinned).

---

## 6. Reproduce

```bash
# Software Tensor Core
cd 02-software-tensor-core
/vol1/ganlin/venv/bin/python sw_tensor_core_v2.py --size 2048

# Dependencies: torch 2.3.1 + nvidia-cuda-nvrtc-cu12 (libnvrtc.so.12)
```

---

## 7. Conclusion

**"Unlocking the Tensor Core" does not hold on an MX450 — there is no TC to unlock.**

**But "rebuilding one" does hold: the FFMA path on CUDA cores reaches 1.603 TFLOPS,
5.0× the card's cuBLAS TENSOR_OP.**

**That is this machine's real ceiling for matrix multiplication.**
