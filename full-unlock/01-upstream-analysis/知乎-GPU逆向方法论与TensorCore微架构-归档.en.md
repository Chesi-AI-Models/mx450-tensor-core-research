# Two Zhihu articles on GPU reverse-engineering methodology — digest archive and mapping to this project

> Sources (read 2026-10-20):
> ① "GPU Reverse Engineering Method Summary" https://zhuanlan.zhihu.com/p/2014778262367250116 (edited 2026-03-10)
> ② "Tensor Core Microarchitecture Deep Dive" https://zhuanlan.zhihu.com/p/2028548125518755193
> This file is a structured archive (with all key data tables and the paper/tool lists), annotated with how each
> item maps onto this project's MX450 work.

---

## 1. Article ① "GPU Reverse Engineering Method Summary" — eight method classes

| Section | Method | Means | What it yields |
|---|---|---|---|
| 0x01 | microbenchmarking | `clock`/`clock64` counting + controlled variables | pipeline latency/throughput, scheduling policy, TC characteristics |
| 0x02 | ISA binary analysis and patching | bit-flipping, SASS patching | instruction encoding format, control-code semantics, dependency barriers |
| 0x03 | memory hierarchy probing | pointer chasing | cache size/associativity/replacement policy, TLB levels, latency |
| 0x04 | binary instrumentation | NVBit / SASSI | runtime instruction-level behaviour |
| 0x05 | simulator-driven | GPGPU-Sim / Accel-Sim | make assumptions explicit, hardware-vs-model comparison |
| 0x06 | side channels and security | timing / contention | scheduler granularity, shared-resource boundaries |
| 0x07 | power and thermal modelling | microbenchmarks + regression | component-level power, power-gating granularity |
| 0x08 | tools and assemblers | CuAssembler / maxas / TuringAs / DocumentSASS | SASS-level modification and optimisation |

### 1.1 Key papers and data (by generation)

| Paper | Architecture | Key findings |
|---|---|---|
| Wong 2010 (ISPASS) | GT200 | three-level instruction cache 4/8/32 KB; L1 TLB 16-way fully associative over 8MB, 512KB per entry; L2 TLB 32MB 8-way, 4KB per entry |
| Jia 2018 (arXiv:1804.06826) | Volta V100 | L2 6144 KiB / 900 GB/s @877MHz; L2 physically indexed; L1.5 constant cache shared with L1I; **6 dependency barriers** |
| Jia 2019 (arXiv:1903.07486) | Turing T4 | instruction encoding identical to Volta; **dual-ported register banks**; **reuse flag covers at most 4 operands per instruction**; L2 16-way; **L1 miss/L2 hit = 188 cycles, L2 miss/TLB hit = 296, full miss = 616** |
| Abdelkhalik 2022 (arXiv:2208.11174) | Ampere A100 | CPI catalogue for all instruction types; **SASS barrier ≈ 33 cycles** |
| Arafa 2019 (arXiv:1905.08778) | 5 generations, 7 cards | low-overhead instruction-latency characterisation framework (NMSU-PEARL/GPUs-ISA-Latencies) |
| Luo 2024 (arXiv:2402.13499) | Hopper H100 | first characterisation of **WGMMA**; **average L2 latency = 6.5× L1** |
| Luo 2025 (arXiv:2501.12084) | Hopper | TMA 1.5×; FP8 ≈ 2× FP16; DPX ≥ 4.75× |
| Huerta 2025 (arXiv:2503.20481, MICRO) | Ampere/Turing/Blackwell | **CGGTY scheduling policy, no traditional scoreboard, no operand collector**; 1024-bit read port per bank; RFC 3×1024-bit per entry; **41 control bits/warp = 0.09% of the register file**; Accel-Sim MAPE 33% → 13.45% |
| Jarmusch 2025a (arXiv:2512.02189) | Blackwell B200 | ResNet-50 1.85× vs H200; cache-miss latency −58%; decompression engine 42–462 GB/s; first characterisation of TMEM |
| Jarmusch 2025b (arXiv:2507.10789) | Blackwell GB203 | L2 hit ≈ 358 cycles (fixed); GH100 L2 65MB |
| Mei & Chu 2017 (arXiv:1509.02308) | Fermi/Kepler/Maxwell | **the GPU L1 data cache does not use LRU** → overturns the traditional P-chase assumption |
| Zhang 2023 (CCS) | Turing/Ampere | **MIG does not partition the L3-uTLB**; L3-uTLB 8-way 2 slices, 8-entry victim buffer per slice |
| Sun 2023 (TPDS) | Volta/Turing/Ampere | ThC generations: Volta 8 per SM 4×4×4 FP16 only; Turing adds INT8/INT4/Binary; Ampere 4 per SM 8×4×8 plus BF16/TF32/sparsity |
| Khairy 2019/2020 | Volta / all generations | memory-model error −66×; **Accel-Sim** 80 workloads 1945 kernels, cycle error −79 percentage points |
| Villa 2019 (NVBit) / Stephenson 2015 (SASSI) | multiple | dynamic / static instrumentation frameworks |
| Leng 2013 / Kandiah 2021 | GPUWattch / AccelWattch | error 9.9%/13.4%; AccelWattch Volta MAPE 7.5–9.2%; **power-gating granularity: L2 at chip level / L1 at SM level / FPU at lane level** |
| Naghibijouybari 2017/2018 | Maxwell/Volta | covert channel > 4 Mbps; side channel led to **CVE-2018-6260** |

### 1.2 ★ Tool list (directly usable on the MX450)

| Tool | Repository | Use for us |
|---|---|---|
| **CuAssembler** | `cloudcores/CuAssembler` | Kepler–Hopper SASS disassembly/assembly/patching → **change the stall/barrier/reuse control codes of our own GEMM** |
| **TuringAs** | `daadaada/turingas` | **SASS assembler for Volta/Turing**, directly matching sm_75 |
| **maxas** | `NervanaSystems/maxas` | Maxwell SGEMM reached **98% of theoretical peak**, **4.8% faster than cuBLAS** → the template for our 80.9% → ~95% |
| **DocumentSASS** | `0xD0GF00D/DocumentSASS` | `LD_PRELOAD` intercepts nvdisasm and extracts the **embedded instruction specs and scheduling latencies** → saves microbenchmarking fixed latencies |
| **nv_isa_solver** | `kuterd/nv_isa_solver` | fuzzes nvdisasm to auto-generate the ISA spec (already produced SM90a/SM89) |
| **AmpItUp** | `NoxNode/AmpItUp` | bit-level annotation of Ampere instruction encoding (method applicable to sm_75) |
| **NVBit / SASSI** | `NVlabs/NVBit` / `NVlabs/SASSI` | instrument our own kernels |
| **Accel-Sim** | `accel-sim/accel-sim-framework` | cross-check microbenchmark conclusions |
| **gpu-tlb** | `0x5ec1ab/gpu-tlb` | TLB reverse engineering (including pagemap/page-table tools) |
| **CuAsmRL / SIP** | arXiv:2501.08071 / 2403.16863 | automatic SASS scheduling search, measured **+9% (up to 26%)** |
| JEB 5.31+ | commercial | Volta–Blackwell decompilation |

### 1.3 Methodology points that can be adopted directly

1. **The CuAsmRL lesson**: clock-based SASS microbenchmarks **can be wrong** (it measured IADD3 as 2.6 cycles);
   a **dependency-based method** (progressively lowering the stall count until the result becomes wrong) is more
   reliable. → We should switch to that method when measuring latency.
2. **Control codes are the hidden layer of performance**: cuBLAS's advantage comes mainly from SASS-level pipeline
   management, not from the algorithm (Schuetze 2024, from reverse-engineering cuBLAS).
3. **TuringAs's key optimisation**: interleave LDG across 8 FFMAs (cuDNN uses only 2) → **1.24×** on that item.
   This matches the direction of our v7 "double buffering + interleaved loads", but it is precise SASS-level control.
4. **NVIDIA patents are an intelligence source for reverse engineering**: `US20150220341A1` (software scoreboard
   DEPBAR, 6-bit counter 0–63), `US11080051B2` (LDGSTS async copy + LDGDEPBAR).

---

## 2. Article ② "Tensor Core Microarchitecture Deep Dive" — ★ directly explains our measurements

### 2.1 Basic facts

- Volta SM = 4 sub-cores; per sub-core: 1 warp scheduler (1 instr/clk), 1 dispatch, 64KB RF, **2 Tensor Cores**,
  8 FP64, 16 FP32, 16 INT, 4 MUFU, MIO, L0/L1 ICache
- Titan V: 80 SM × 8 TC = 640 TC; 1530 MHz → **theoretical 125 TFLOPS (FP16 mixed precision)**
- PTX `wmma.mma` → multiple **HMMA** SASS instructions

### 2.2 ★★ The SET/STEP structure of HMMA.884 (directly applicable to our MX450)

```
HMMA.884.F32.F32.STEP0 R8,  R24.reuse.COL, R22.reuse.ROW, R8;   ← cycle 10
HMMA.884.F32.F32.STEP1 R10, R24.reuse.COL, R22.reuse.ROW, R10;  ← cycle 12
HMMA.884.F32.F32.STEP2 R4,  R24.reuse.COL, R22.reuse.ROW, R4;   ← cycle 14
HMMA.884.F32.F32.STEP3 R6,  R24.COL,       R22.ROW,       R6;   ← cycle 18
```

| Fact | Value |
|---|---|
| one `wmma.mma` 16×16×16 mixed precision → **16 HMMA instructions** (4 SET × 4 STEP) | accumulates over 44 cycles (10→54) |
| FP16 accumulate mode → **8 HMMA** (4 SET × 2 STEP) | total latency 64 cycles |
| a single register number = **a pair of adjacent registers** | R8 means R8:R9 |
| the `reuse` annotation = cached into the **Operand Reuse Cache** | at most 4 operands per instruction |
| per SET: 4×4 A × 4×8 B → 4×8 C/D; per STEP: 2×4 A × 4×4 B | 4×4 = 16 basic multiply-adds |
| per TC = 16 FEDP units, 4-stage pipeline | register bandwidth 6 kb/2 cycles/warp = 1.5 kb/cycle/TC |
| **TC cannot co-issue with INT/FP32** | because it saturates the register access ports (3 sources × 512 bit = 1536 bit/cycle) |
| concurrency | **only 4 warps per SM can execute HMMA concurrently**; each warp uses 2 TCs |

### 2.3 Turing's timing table (Table I, 16×16×16)

| Precision | SET1 | SET2 | SET3 | SET4 | Total latency |
|---|---|---|---|---|---|
| FP32 accumulate | 42 | 56 | 78 | 99 | **99 cycles** |
| FP16 accumulate | 44 | 52 | 60 | 74 | 74 cycles |
| INT8 | 40 | 44 | 47 | 59 | 59 cycles |

**Turing's 16×16×16 mixed precision needs 99 cycles, slower than Volta's 54** (but Turing supports larger tiles:
32×8×16 / 8×32×16 / 8×8×32).

### 2.4 ★★★ The decisive inference for this project

**What we measured**: on the MX450 (TU117, sm_75) cuBLAS fp16 takes `cutlass_70_tensorop_f16_s884gemm` and reaches
only **0.3191 TFLOPS = 9.9% of the clock-normalised peak** (FP32 FFMA reaches 99.0%).

**The mechanism article ② provides**:
1. `s884` is exactly the **`mma.m8n8k4` → `HMMA.884`** path; one wmma needs 16 HMMA instructions and 99 cycles;
2. but **TU117 has no physical Tensor Core** (this project measured it: FP16 5.77 vs mma 0.60, 9.6× slower),
   so every HMMA.884 must be **emulated one by one** on the ordinary FP16/FEDP path;
3. and HMMA **cannot co-issue with FP32/INT** (it occupies all register ports),
   → during emulation you get neither TC throughput nor scalar pipeline progress → only ~10% remains.

**→ That is the complete causal chain for the 9.9%, not "the driver is unoptimised" but missing hardware plus a
mismatched instruction sequence.**

**It follows that**: on the MX450 any "library that takes the HMMA path" (cuBLAS fp16 / Triton / TileLang /
FlagGems) hits the same wall (measured Triton 0.595 / TileLang 0.538 / FlagGems 0.29-0.55×, all corroborating).
**The only way out is our own FFMA path** (recorded then as 2.4872 TFLOPS; see the translator's note).

---

## 3. Mapping both articles onto this project's next steps (actionable)

| Article method | What we have done | Direct upgrade action |
|---|---|---|
| 0x01 microbenchmarks | FMA peak 2.4747 (99.0%), GEMM 2.4872 (80.9%), clock sampling | keep; add the **dependency-based method** (the CuAsmRL lesson) |
| 0x02 SASS analysis/patching | captured the kernel name `cutlass_70_tensorop_f16_s884gemm` | **adopt CuAssembler / TuringAs to change the stall-count / barrier / reuse of our own GEMM** |
| 0x03 memory hierarchy | DRAMA++ / Sudoku for DRAM bank mapping | add **pointer-chasing measurement of MX450 L1/L2/TLB** (doable within 2GB VRAM) |
| 0x04 instrumentation | none | use NVBit to instrument our own GEMM and see why it stalls |
| 0x05 simulators | none | use Accel-Sim to check whether our 80.9% is already the ceiling for this configuration |
| 0x07 power/thermal | only read clocks_event_reasons | **AccelWattch approach: infer power gating from microbenchmarks** (our power.draw is N/A, so only indirect means) |
| 0x08 tools | none | **DocumentSASS to extract nvdisasm's embedded latency table → know each sm_75 instruction's exact latency** |

**Highest priority**: 0x08 DocumentSASS + 0x02 CuAssembler/TuringAs.
Rationale: on Maxwell, maxas used exactly this method to reach **98% of theoretical peak, 4.8% above cuBLAS**;
we are at 80.9%, and what we lack is precisely **precise SASS-level stall/barrier control** (not the algorithm,
not the tile size).

---

## Translator's note (2026-10-23)

This archive is a faithful record of what was read and measured on 2026-10-20, and is **not rewritten**. However,
a later reliable benchmark (interleaved 12 rounds, median, `reliable_bench.py`) showed that the "2.4872 TFLOPS /
80.9%" figures quoted above came from **single-run best-of-N measurements**, which suffer up to 15% run-to-run
drift on this machine. The median-based result is:

| Implementation | Median TFLOPS |
|---|---|
| self-built FFMA GEMM | **1.50** |
| cuBLAS fp32 | **2.06** |

That is, the self-built kernel is **73% of cuBLAS**, not above it. The microarchitectural conclusions in this
document (HMMA emulation chain, TC cannot co-issue, the FFMA path being the only way out) are **unaffected** —
only the absolute throughput numbers need this correction.
