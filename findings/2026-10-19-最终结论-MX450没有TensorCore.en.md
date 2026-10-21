# 🔴🔴🔴 Final conclusion: the MX450 has no Tensor Core

## 1. I must first admit a judgement error

**Throughout this project the stated position was: "Tensor Cores exist, but are throttled to 5% of theoretical
(0.60 vs 6–12 TFLOPS)."**

**That judgement was wrong.**

**The correct conclusion is: the SM of TU117 contains no Tensor Core hardware.
0.60 TFLOPS is the *emulated* performance of FP16 units executing HMMA instructions — not a "throttled TC".**

**This error persisted through the entire project cycle and was only corrected on 2026-10-19 14:30 via a web search.**

## 2. Confirmation from public sources (three independent)

### Expreview review, 2019-04-24, original text:

> "The TU117 used by GTX 1650, like TU116 before it, **has RT Core and Tensor Core removed**;
> the SM internal architecture was modified — **the location previously holding Tensor Cores became FP16
> half-precision floating-point units**, which process game data more efficiently than the older Pascal architecture"

### Notebookcheck (GTX 1650 Laptop):

> "Compared to the faster RTX 2000 GPUs (e.g. RTX 2060), **the 1650 has no integrated Raytracing or Tensor cores**"

### Baidu Baike (GeForce GTX 1650):

> "The core of the GTX 1650 (TU117) **removes the RT Core dedicated to real-time ray tracing and the Tensor Core
> used for AI supersampling (DLSS)**"

### Control group: RTX 20 series (TU10x)

> "Each SM contains 64 CUDA cores, **8 Tensor Cores**, 256 KB of registers, and 4 texture units"

**TU117: 0 Tensor Cores. That area is taken by FP16 units.**

## 3. Measured evidence chain (local machine)

### Experiment 1 · FP32 / FP16 / mma three-way comparison

```
GPU: NVIDIA GeForce MX450   SM=14   capability=7.5
matrix 2048×2048, 17.2 GFLOP per call, 20 iterations

FP32 (allow_tf32=False) : 2.122 TFLOPS   ← theoretical 2.5, 85% reached, methodology sound
FP16 (PyTorch)          : 0.327 TFLOPS   ← only 15% of FP32
FP16 / FP32 = 0.15x
```

### Experiment 2 · compared against historical measurements

| Path | Measured TFLOPS |
|---|---|
| FP32 cuBLAS (`volta_sgemm_128x64_nn`) | **2.122** |
| hand-written PTX `mma.sync` | **0.60** |
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | **0.32** |
| **PyTorch FP16 CUDA core** | **0.327** |

**Key: cuBLAS's "TENSOR_OP" path (0.32) is almost identical to FP16 on CUDA cores (0.327).**

**This shows that the so-called "TENSOR_OP" path is running on FP16 units.**

### Experiment 3 · kernel name (decisive)

**The cuBLAS log confirms it selected a TC algorithm:**
```
algo: type=SOME TYPE; val=CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)
```

**But the kernel actually executed was:**
```
FP16: void cutlass::Kernel<cutlass_70_tensorop_f16_s884gemm_relu_f16_1...
                            ↑ cutlass_70          ↑ s884 = m8n8k4
FP32: volta_sgemm_128x64_nn
```

**Two critical points:**

1. **`cutlass_70` = compiled for SM 7.0 (Volta), not sm75**
   → NVIDIA **never provided an sm75 TC kernel for TU117**
   → cuBLAS falls back to the generic sm70 kernel

2. **`s884` = the m8n8k4 mma shape** (Volta's first-generation TC shape)
   → not Turing's m16n8k8 / m16n8k16

**Performance: 276,944 µs / 5 = 55 ms per matmul → 0.31 TFLOPS**

## 4. The complete explanation

**Why do HMMA instructions execute at all?**

CUDA capability 7.5 declares support for the full sm_75 instruction set, **including HMMA**.
The instructions compile, decode, and produce correct results.

**But the execution unit is not a TC.**

In TU117's SM, the area that would hold TCs holds **FP16 units**.
HMMA instructions are mapped onto FP16 units — **the results are correct, the speed is FP16's**.

**This explains three things:**

| Observation | Explanation |
|---|---|
| HMMA compiles, executes, results correct | instruction-set support |
| performance only 0.31–0.60 TFLOPS | executed by FP16 units, not TCs |
| cuBLAS picks cutlass_70 rather than sm75 | NVIDIA never shipped an sm75 TC kernel for TU117 |
| FP16 CUDA core ≈ TENSOR_OP performance | both use the same execution unit |

## 5. Why every earlier "lock hunt" failed

**Because this project was hunting for something that does not exist.**

Reviewing the road travelled:

| Stage | What was done | Result |
|---|---|---|
| hardware measurement | measured 0.60 TFLOPS, judged "throttled to 5%" | **the premise itself was wrong** |
| GSP firmware | full disassembly of a 30 MB RISC-V ELF | contains no TC logic (as expected — TCs do not exist) |
| vBIOS | found FWSEC certificate-chain verification | unrelated to TC |
| system BIOS | Boot Guard encrypted | unrelated to TC |
| PRAMIN | opened VRAM read/write, dumped 32 MB | no TC configuration in VRAM |
| FECS microcode | full 32 KB dump + 11,401 lines disassembled | handles only scheduling and synchronization |
| GPCCS microcode | full 16 KB dump + 6,059 lines disassembled | same as above |
| Falcon IO | confirmed the host window is only 4 KB | the `0x20000` segment is invisible |
| nvidia.ko | symbol-level RE, 46,614 symbols | all five clue classes excluded, zero hits for Tensor/MMA |

**Nine paths, all failed — because they were all looking for a "lock", while the problem is that there is no door.**

**In hindsight the most convincing evidence was there from the very start:
`tensor|HMMA|MMA_` string searches in `nvidia.ko` returned zero hits.
If a card had TCs, the driver could not possibly contain not a single related symbol.**

## 6. What this conclusion actually means

### For the goal of "unlocking the TC"

**Impossible.**
Not "locked and unopenable", but "this function does not exist in silicon".

**The only way to get TCs is to change cards.**

### For the MX450's real positioning

| Item | Real value |
|---|---|
| FP32 | 2.12 TFLOPS (theoretical 2.5, 85% reached) |
| FP16 (emulated) | 0.33–0.60 TFLOPS |
| **Tensor Core** | **none** |
| RT Core | none |
| VRAM | 1.68 GB usable / 2 GB physical |
| 7B QLoRA requirement | ~4 GB |

**The MX450 is a fully specified entry-level FP32 graphics card, and its FP32 efficiency (85%) is actually quite
good. It simply has no TC, and cannot have one.**

### For the value of this project

**This research still has value, but its positioning has changed:**

It is not "I tried to unlock the TC and failed",
but "**I proved with nine paths that TU117 has no TC, and left behind a complete evidence chain and methodology**".

**That includes:**

- a reproducible Turing reverse-engineering toolchain (PRAMIN, Falcon IMEM dump, nvidia.ko symbol analysis)
- complete dumps and disassembly of FECS/GPCCS microcode (not seen in public material)
- a complete model of the Falcon IO address space (`HOST_IO_INDEX` is only 6 bits)
- the GRCTX layout of nvidia.ko (all 186 fields located)
- **an important methodology: judging *whether something exists* matters more than judging *whether it can be
  enabled*, and should be done earlier**

## 7. Methodology lessons (written to myself)

**The biggest lesson of this project: before investing large effort in finding an "unlock method",
first confirm — with the cheapest possible method — whether the thing exists at all.**

**What is the cheapest method?**
1. check the public specifications (30 seconds) — Expreview wrote it clearly back in 2019
2. check whether the driver contains related symbols (1 minute) — `strings nvidia.ko | grep -i tensor` returns nothing
3. compare the performance magnitude against similar hardware (5 minutes)

**Those three steps take under 10 minutes and would have saved 20 hours of the entire project.**

**We instead discovered it after 20 hours, through a single web search.**

**That is the real value of "search the web at every step" — it is not merely supplementary information,
it prevents you from building an entire castle on a false premise.**

## 8. Final data summary

```
device: NVIDIA GeForce MX450 (TU117)
SM: 14    capability: 7.5    VRAM: 2048 MiB (1681 MiB usable)
driver: 580.142    CUDA: 13.0

FP32 peak:  2.122 TFLOPS  (volta_sgemm_128x64_nn)
FP16 peak:  0.327 TFLOPS  (CUDA core)
mma.sync:   0.600 TFLOPS  (hand-written PTX)
cuBLAS TC:  0.320 TFLOPS  (cutlass_70_tensorop_f16_s884gemm)

Tensor Core: does not exist (removed on TU117; that area holds FP16 units)
```

## 9. Acknowledgements and statement

**This conclusion rests on:**
- 3 independent public sources (Expreview / Notebookcheck / Baidu Baike)
- 4 local measurement groups (FP32 / FP16 / mma.sync / cuBLAS TENSOR_OP)
- 1 cuBLAS kernel-name capture (`cutlass_70_tensorop_f16_s884gemm`)
- exclusionary evidence from 9 reverse-engineering paths

**All measurements are reproducible with `tools/bench_fp16_vs_tc.py` and `tools/profile_kernel2.py`.**

**Any earlier statement in these findings that "the TC is throttled" is superseded by this document.**
