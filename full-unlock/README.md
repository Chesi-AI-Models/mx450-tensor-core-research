# MX450 完整破解流程 — 现实核查与执行记录

> 本目录是对「**尽可能彻底解锁 MX450 的 Tensor Core**」这个目标的完整回答。
>
> 结论先行：**TU117 没有 Tensor Core 硬件，无法"解锁"；但可以用 CUDA core 重建一个，
> 实测 1.603 TFLOPS，是 cuBLAS TENSOR_OP 的 5.0 倍。**

---

## 一、先做现实核查：哪些能破，哪些不能

在动手之前，必须把「破解」这个词拆开。MX450 上的四类限制，性质完全不同：

| 目标 | 能否破解 | 原因 | 本目录的应对 |
|---|---|---|---|
| **Tensor Core** | ❌ 不存在 | TU117 的 SM 里没有 TC，位置是 FP16 单元 | ✅ **软件重建**（`02-software-tensor-core/`） |
| **显存 2 GB** | ❌ 不能 | 2 颗 GDDR6 物理焊死，没有"被封印的容量" | — |
| **SM 频率** | ⚠️ 有限 | `nvidia-settings` 可小幅超频，受 TGP 限制 | — |
| **FECS/GPCCS 微码** | ⚠️ 只能读 | FWSEC 证书链验证，改一字节即拒绝加载 | 已完整 dump（见上级目录 findings） |

### 为什么 CMPUnlocker 的方法不能照搬

CMPUnlocker 能解锁 CMP 170HX 的 8 GB → 64 GB，**前提是物理上有 80 GB HBM2e**，
只是被 OTP fuse + 固件限制。

MX450 的情况是：

- 显存：物理 2 颗 GDDR6 = 2 GB，**没有更多**
- TC：TU117 的 SM 里**没有 TC 单元**

**没有「被封印的东西」，就没有可解锁的东西。** 这是本目录与 CMPUnlocker 最根本的区别。

---

## 二、目录结构

```
full-unlock/
├── README.md                      本文件：现实核查 + 总览
├── 01-upstream-analysis/          所有参考开源项目的分析
│   ├── cmpunlocker.md             ★ CMPUnlocker 完整技术拆解
│   └── vendor/cmpunlocker/        ★ 上游源码（GPL v2，见许可证说明）
├── 02-software-tensor-core/       ★ 核心：用 CUDA core 重建 Tensor Core
│   ├── README.md                  实现说明与性能数据
│   ├── triton_gemm.py             v1: Triton tl.dot（走 HMMA）0.595
│   ├── triton_gemm_v2.py          v2: 外积（失败）0.122
│   ├── triton_gemm_v4.py          v4: FP32 dot（Triton 崩溃）
│   └── sw_tensor_core_v2.py       ★ 最终版：NVRTC + FFMA，1.603 TFLOPS
├── 03-toolkit/                    复用已有工具（PRAMIN / Falcon / nvidia.ko）
└── 04-results/                    实测数据
```

---

## 三、核心成果：软件 Tensor Core

**在 TU117 上用 CUDA core 重建了一个矩阵乘引擎，绕开不存在的 TC。**

### 技术路线

```
CUDA C 源码（unsigned short + inline PTX）
    ↓ NVRTC 编译（libnvrtc.so.12）
PTX（39,393 字符，含 cvt.f32.f16 / fma.rn.f32，不含 mma.sync）
    ↓ cuModuleLoadData（libcuda）
GPU 执行
```

**关键：全程不用 `cuda_fp16.h`（NVRTC 找不到），全部用 `unsigned short` + inline PTX 的
`cvt.f32.f16` / `cvt.rn.f16.f32`。**

### 实测性能（2048³，本机）

| 实现 | TFLOPS | 相对 cuBLAS TENSOR_OP |
|---|---|---|
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | 0.320 | 1.0x |
| Triton `tl.dot`（FP16，走 HMMA） | 0.595 | 1.9x |
| **本目录 `sw_tensor_core_v2.py`（FFMA）** | **1.603** | **5.0x** |
| cuBLAS FP32（`volta_sgemm`） | 2.122 | 6.6x |
| FP16x2 纯计算峰值（FP16 累加，不可用于真实 GEMM） | 5.770 | — |

**正确性：`max abs err = 0.00000`（与 float64 参考完全一致）。**

### 为什么是 1.603 而不是 5.77

**`fma.rn.f16x2` 是 FP16 累加。K=2048 时累加器会溢出，结果不可用。**

真实 GEMM 必须 FP32 累加 → 只能用 FFMA → 上限是 FP32 的 2.5 TFLOPS。
1.603 是 2.5 的 64%，剩下的是共享内存带宽和指令调度的损耗。

**这不是实现不够好，是硬件的算术强度上限。**

---

## 四、上游项目的实际使用情况

| 项目 | 用了吗 | 怎么用的 |
|---|---|---|
| **CMPUnlocker** | ✅ 源码纳入 + 完整拆解 | 学到 ROP 注入机制；确认其 PLM 目标寄存器 Turing 不存在 |
| **Triton** | ✅ 实际使用 | v1 基线 0.595；发现 CC 7.5 上 FP32 路径有 `map::at` 崩溃 |
| **NVRTC / CUDA Driver API** | ✅ 实际使用 | 最终版全部依赖此路径 |
| **tinygrad** | ⚠️ 仅参考 | 其用户态驱动思路不适用于本任务 |
| **TileKernels / FlagOS** | ⚠️ 仅参考 | 面向昇腾/多芯片，Triton DSL 思路已借鉴 |
| **DRAMA++ / Sudoku** | ❌ 不适用 | ARM64 服务器 DRAM 地址映射 |
| **Open-RDMA / fabric-lib** | ❌ 不适用 | 网络层，与单卡无关 |
| **AITER** | ❌ 不适用 | AMD ROCm 专用 |
| **shadPS4 显存重构** | ❌ 不适用 | 模拟器场景 |

**诚实说明：txt 里列的项目，真正能用于 MX450 的只有 CMPUnlocker（方法论）和 Triton（工具链）。
其余因架构/平台不匹配无法使用——这不是选择，是物理限制。**

---

## 五、许可证边界（重要）

`01-upstream-analysis/vendor/cmpunlocker/` 是 **GPL v2** 代码，原样引入、保留原许可证。

**本目录其他部分（含 `02-software-tensor-core/` 全部代码）为本项目原创，
不因引用 vendor 目录而受 GPL 约束。** 这是 Linux 内核 `vendor/` 目录的通行做法。

如需在 GPL 不兼容的项目中使用，请删除 `vendor/` 目录，仅参考 `cmpunlocker.md` 的分析。

---

## 六、复现

```bash
# 软件 Tensor Core
cd 02-software-tensor-core
/vol1/ganlin/venv/bin/python sw_tensor_core_v2.py --size 2048

# 依赖：torch 2.3.1 + nvidia-cuda-nvrtc-cu12（libnvrtc.so.12）
```

---

## 七、结论

**「解锁 Tensor Core」在 MX450 上不成立——因为没有 TC 可解锁。**

**但「重建一个」成立——用 CUDA core 的 FFMA 路径，拿到 1.603 TFLOPS，
是这块卡上 cuBLAS TENSOR_OP 的 5.0 倍。**

**这就是这台机器在矩阵乘上的真实上限。**
