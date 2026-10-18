# MX450 (TU117) Tensor Core Reverse Engineering

[简体中文](README.md) | [繁體中文](README.zh-TW.md) | **English**

> Investigating why the NVIDIA GeForce MX450 (TU117M) throttles its Tensor Cores to 5–10% of
> theoretical throughput, and systematically mapping every accessible boundary in this machine's
> firmware and driver stack.
>
> **Status: all publicly-documented boundaries explored · no unlock achieved · conclusions are evidence-based**

---

## One-line conclusion

**This card genuinely has a locked compute space — Tensor Core instructions exist, compile, and
execute, but peak at 5–10% of theoretical throughput. The machine seals it with five layers;
this research verified and bypassed four of them. The fifth is register semantics NVIDIA has
never published.**

---

## 1. Starting point: an anomalous measurement

| Path | Measured | Notes |
|---|---|---|
| Hand-written PTX `mma.m16n8k8` (Tensor Core instruction) | **0.60 TFLOPS** | Tensor path |
| Hand-written PTX `FP16x2` (CUDA cores, no TC) | **5.77 TFLOPS** | Pure FP16 units |
| Hand-written PTX `FP32 FMA` | 3.06 TFLOPS | Baseline |
| `cuBLAS` FP16, log confirms `CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)` | **0.32 TFLOPS** | NVIDIA's own library |
| `cuBLAS` FP32 (no TC) | 2.07 TFLOPS | Control |
| TU117 theoretical tensor peak (14 SM × 4 TC) | 6 – 12 TFLOPS | — |

**TC instructions exist, compile, execute without error — yet run 5–6× slower than not using TC.**

---

## 2. Five layers of lockdown and their outcomes

| # | Layer | Conventional wisdom | This research |
|---|---|---|---|
| ① | **Hardware**: GPU has no dedicated SPI flash | vBIOS not flashable | ❌ Not bypassable (physical) |
| ② | **System firmware**: Intel Boot Guard | BIOS not parseable | ❌ Not bypassable (needs OEM key) |
| ③ | **GPU firmware**: GSP FWSEC cert chain | Microcode not modifiable | ❌ Not bypassable (needs NVIDIA key) |
| ④ | **Driver layer**: NVML / CUPTI / private ioctl | Low-level access blocked | ✅ **Partially bypassed** (PRAMIN) |
| ⑤ | **Register semantics**: GPC/TPC offsets unpublished | — | ⚠️ **Unresolved** (info does not exist) |

---

## 3. Twenty-one established facts

| # | Fact | Evidence |
|---|---|---|
| 1 | TC exists but is throttled to ~5% | Hand-written PTX 0.60 vs theoretical 6–12; cuBLAS agrees |
| 2 | GSP firmware contains no TC control logic | Full disassembly; 8 Ampere register addresses absent |
| 3 | vBIOS is FWSEC cert-chain protected | 8 `VERIFY_*` error codes found in firmware |
| 4 | BAR1 "16 MB only" is bypassable | PRAMIN channel, write-readback verified |
| 5 | P2P "consumer EINVAL" is bypassable | Avoid P2P entirely |
| 6 | BAR0 is accessible while driver runs | Kernel module `ioremap` |
| 7 | VRAM is readable | 32 MB dump, MD5 verifiable |
| 8 | VRAM does not contain Falcon microcode | Kepler signature: zero hits |
| 9 | FECS Falcon base = `0x409000` | Measured |
| 10 | `HWCFG1` is at `+0x12c` (not `+0x10c`) | Source-confirmed |
| 11 | **FECS runs in LIGHT SECURE mode** | `security_model = 2` (not HS black box) |
| 12 | **FECS microcode is fully readable** | 32 KB captured in full |
| 13 | Microcode shares structure with gk20a | 7-byte signature `00fe0400bda47e` matches |
| 14 | Entry = set stack top + `lcall main` + `exit` | Disassembly |
| 15 | ISA is fuc6 | `core_rev = 6` |
| 16 | 449 register writes, 100% of addresses resolved | Static analysis |
| 17 | 379 writes land in `0x20000–0x2FFFF` | Segment statistics |
| 18 | **Semantics of that segment unknown** | All three routes failed |
| 19 | PGRAPH lives at `0x400000` | envytools rnndb (authoritative) |
| 20 | Microcode cannot be modified | FWSEC signature verification |
| 21 | 1.68 GB VRAM is immutable | Physically soldered |

---

## 4. Key breakthrough: reading the FECS microcode

Five locks, opened one by one:

| Lock | Conventional wisdom | Bypass |
|---|---|---|
| BAR1 maps only 16 MB | VRAM unreadable | Use **PRAMIN** (`BAR0+0x700000`) |
| P2P returns EINVAL on GeForce | Cannot pin VRAM | Do not use P2P |
| BAR0 busy while driver loaded | Registers unreachable | **Kernel module `ioremap`** |
| Falcon in HS = black box | Completely unreadable | **Measured: LS** (critical) |
| Falcon `halted=1` when idle | IMEM all zeros | **Force GPU to 100% load** |

**Read sequence:**

```c
// 1. Keep GPU at 100% load (torch GEMM loop)
// 2. insmod mx450_p4.ko bar0_phys=<BAR0> mode=dump dump_bytes=32768
// 3. Inside the module:
//    ioremap(BAR0 + 0x409000, 0x1000)
//    Read HWCFG1, verify security_model == 2 (LS)
//    Save IMEMC(0)
//    for (addr = 0; addr < 32768; addr += 4) {
//        blk  = addr / 256;
//        word = (addr % 256) / 4;
//        writel((blk<<8)|(word<<2), base + 0x180);   // IMEMC
//        *(u32*)(buf+addr) = readl(base + 0x184);    // IMEMD
//    }
//    Restore IMEMC(0)
```

---

## 5. Artifacts

| Artifact | Size | Verification |
|---|---|---|
| `fecs_imem.bin` — raw FECS microcode | 32 KB | MD5 `ea8eb3c46165b00421896eef45d1070a` |
| `fecs_plain.asm` — full disassembly | 1.3 MB / 11401 lines | — |
| `vram_dump.bin` — VRAM dump | 32 MB | MD5 `1b224dcfeec507673ee64fd35d44005c` |
| `bios_readonly.bin` — system BIOS | 24 MB | MD5 `4d2c99e85e761e0a3b1b8dcada4cc950` |
| `envytools/envydis` — disassembler | — | Build locally |

> Binary artifacts are excluded from the repo due to size; regenerate them with the scripts in `tools/`.

---

## 6. Repository layout

```
.
├── README.md                    Simplified Chinese
├── README.zh-TW.md              Traditional Chinese
├── README.en.md                 English (this file)
├── LICENSE
├── docs/
│   ├── 02-技术路线图.md
│   └── 03-内核模块路线图.md
├── findings/                    23 append-only field reports
├── refs/                        NVIDIA public headers (MIT, verbatim)
├── kernel/                      6 kernel modules
│   ├── mx450_probe.c            toolchain probe
│   ├── mx450_p1.c               kprobe symbol resolution
│   ├── mx450_p2.c               no-fault kernel memory read
│   ├── mx450_p3.c               PRAMIN VRAM access
│   └── mx450_p4.c               FECS IMEM read
├── tools/                       20 analysis tools
└── scripts/                     QEMU isolated test, etc.
```

---

## 7. Test environment

| Item | Value |
|---|---|
| Machine | LYLNAS (Lenovo Zhaoyang E4-ITL modified) |
| OS | fnOS 1.2.0505 / Debian 12 / kernel 6.18.18 |
| Board / BIOS | LNVNB161216 / F8CN59WW(V2.22) 2024-06-14 |
| CPU | Intel i7-1165G7 (4C8T) |
| GPU | GeForce MX450, TU117M, `10de:1f97`, `NV_PMC_BOOT_0 = 0x167000a1` |
| VRAM | 2048 MiB GDDR6, 64-bit, single-allocation ceiling 1620 MiB |
| Driver | 580.142 (proprietary) / CUDA 13.0 |
| Toolchain | gcc 12.2 / g++ 12.2 / cmake 3.25 / envytools / flashrom / strace |

---

## 8. Reproduce

```bash
# Build kernel modules
cd kernel && make

# Read FECS microcode (GPU must be under load)
insmod mx450_p4.ko bar0_phys=<BAR0> mode=dump dump_bytes=32768

# Disassemble
envydis -m falcon -V fuc6 -i -n fecs_imem.bin > fecs_plain.asm

# Analyze register accesses
python3 tools/analyze_iowr.py
```

---

## 9. The unresolved boundary

**The semantics of the `0x20000` segment could not be determined.** Three routes attempted:

| Route | Result |
|---|---|
| envytools rnndb | Coverage ends at Maxwell; recorded upper bound `0x227FF` is below the microcode's actual `0x23800` |
| NVIDIA open headers | `turing/tu102/` has no `dev_gpc.h` / `dev_tpc.h` |
| NVIDIA open HAL | `NV_PGRAPH_GPC` / `NV_PGRAPH_TPC`: zero definitions; logic lives in closed `nv-kernel.o_binary` |

**This is not "not found yet" — it is "NVIDIA has not published this layer."**

---

## 10. Risk disclaimer

1. **Hardware damage is possible.** Writing undocumented values to GPU registers can brick initialization.
2. **Laptop discrete GPUs have no dedicated vBIOS flash.** There is **no firmware region to roll back to**.
3. **No guarantee of success.** This project has established that the microcode is FWSEC-signed: readable, not modifiable.
4. **1.68 GB VRAM is physically soldered.** No software can change it.
5. **Research and educational use only**, on hardware you legally own.

---

## 11. Notice

This project is not affiliated with NVIDIA. All trademarks belong to their respective owners.
The repository contains no proprietary NVIDIA code or firmware; files under `refs/` are NVIDIA
public headers released under the MIT license, reproduced verbatim.

## License

MIT (see `LICENSE`). Findings may be cited freely; use at your own risk.
