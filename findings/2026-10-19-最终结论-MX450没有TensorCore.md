# 🔴🔴🔴 最终结论：MX450 没有 Tensor Core

## 一、我必须先承认一个判断错误

**本项目此前一直的表述是：「Tensor Core 存在，但被限速到理论值的 5%（0.60 vs 6–12 TFLOPS）」。**

**这个判断是错的。**

**正确的结论是：TU117 的 SM 里没有 Tensor Core 硬件。
0.60 TFLOPS 是 FP16 单元执行 HMMA 指令的模拟性能，不是「被限速的 TC」。**

**这个错误持续了整个项目周期，直到 2026-10-19 14:30 通过联网检索才被纠正。**

## 二、公开资料的确认（三方独立）

### 超能网（Expreview）2019-04-24 评测，原文：

> 「GTX 1650 所用的 TU117 和之前的 TU116 一样**去掉了 RT Core、Tensor Core**，
> SM 单元内部架构经过了修改，**原本放 Tensor Core 的位置变成了 FP16 半精度浮点运算单元**，
> 处理游戏数据要比从前的 Pascal 架构来得更高效」

### Notebookcheck（GTX 1650 Laptop）：

> 「与更快的 RTX 2000 GPU（例如 RTX 2060）相比，**1650 没有集成 Raytracing 或 Tensor 内核**」

### 百度百科（GeForce GTX 1650）：

> 「GTX 1650 的核心（TU117）**移除了专门用于实时光线追踪的 RT Core 和用于 AI 超采样（DLSS）的 Tensor Core**」

### 对照组：RTX 20 系列（TU10x）

> 「每个 SM 包含 64 个 CUDA 核心、**8 个 Tensor Cores**、一个 256KB 寄存器、4 个纹理单元」

**TU117：0 个 Tensor Core。位置被 FP16 单元取代。**

## 三、实测证据链（本机）

### 实验 1 · FP32 / FP16 / mma 三路对比

```
GPU: NVIDIA GeForce MX450   SM=14   capability=7.5
矩阵 2048×2048，每次 17.2 GFLOP，迭代 20 次

FP32 (allow_tf32=False) : 2.122 TFLOPS   ← 理论 2.5，达 85%，方法正确
FP16 (PyTorch)          : 0.327 TFLOPS   ← 只有 FP32 的 15%
FP16 / FP32 = 0.15x
```

### 实验 2 · 与历史实测对照

| 路径 | 实测 TFLOPS |
|---|---|
| FP32 cuBLAS（`volta_sgemm_128x64_nn`） | **2.122** |
| 手写 PTX `mma.sync` | **0.60** |
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | **0.32** |
| **PyTorch FP16 CUDA core** | **0.327** |

**关键：cuBLAS 的 "TENSOR_OP" 路径（0.32）与 FP16 CUDA core（0.327）几乎完全相同。**

**这说明：所谓 "TENSOR_OP" 路径，跑的就是 FP16 单元。**

### 实验 3 · kernel 名（决定性）

**cuBLAS 日志确认它选了 TC 算法：**
```
algo: type=SOME TYPE; val=CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)
```

**但实际执行的 kernel 是：**
```
FP16: void cutlass::Kernel<cutlass_70_tensorop_f16_s884gemm_relu_f16_1...
                            ↑ cutlass_70          ↑ s884 = m8n8k4
FP32: volta_sgemm_128x64_nn
```

**两个关键点：**

1. **`cutlass_70` = 为 SM 7.0（Volta）编译，不是 sm75**
   → NVIDIA **没有为 TU117 提供 sm75 的 TC kernel**
   → cuBLAS 回退到 sm70 的通用 kernel

2. **`s884` = m8n8k4 的 mma 形状**（Volta 第一代 TC 形状）
   → 不是 Turing 的 m16n8k8 / m16n8k16

**性能：276944 µs / 5 次 = 55 ms per matmul → 0.31 TFLOPS**

## 四、完整解释

**为什么 HMMA 指令能执行？**

CUDA capability 7.5 声明支持 sm_75 的全部指令集，**包括 HMMA**。
指令能被编译、能被解码、能产生正确结果。

**但执行单元不是 TC。**

TU117 的 SM 里，原本放 TC 的位置是 **FP16 单元**。
HMMA 指令被映射到 FP16 单元执行——**结果是正确的，速度是 FP16 的**。

**这就解释了三件事：**

| 现象 | 解释 |
|---|---|
| HMMA 能编译执行、结果正确 | 指令集支持 |
| 性能只有 0.31–0.60 TFLOPS | 由 FP16 单元执行，不是 TC |
| cuBLAS 选 cutlass_70 而非 sm75 | NVIDIA 没有为 TU117 出 sm75 TC kernel |
| FP16 CUDA core ≈ TENSOR_OP 性能 | 两者走同一个执行单元 |

## 五、为什么之前所有的「找锁」都失败了

**因为这个项目在找一个不存在的东西。**

回顾本项目走过的路：

| 阶段 | 做了什么 | 结果 |
|---|---|---|
| 硬件实测 | 测出 0.60 TFLOPS，判定"被限速到 5%" | **前提就错了** |
| GSP 固件 | 完整反汇编 30 MB RISC-V ELF | 不含 TC 逻辑（合理，因为 TC 不存在） |
| vBIOS | 发现 FWSEC 证书链验证 | 与 TC 无关 |
| 系统 BIOS | Boot Guard 加密 | 与 TC 无关 |
| PRAMIN | 打通显存读写，dump 32 MB | 显存里没有 TC 配置 |
| FECS 微码 | 完整 dump 32 KB + 反汇编 11401 行 | 只管调度与同步 |
| GPCCS 微码 | 完整 dump 16 KB + 反汇编 6059 行 | 同上 |
| Falcon IO | 确认 host 窗口仅 4 KB | `0x20000` 段不可见 |
| nvidia.ko | 符号级逆向，46614 符号 | 五类线索全部排除，Tensor/MMA 零命中 |

**九条路径，全部失败——因为它们都在找"锁"，而问题是"没有门"。**

**现在回头看，最有说服力的证据其实一开始就在：**
**`nvidia.ko` 里 `tensor|HMMA|MMA_` 字符串零命中。**
**如果一块卡有 TC，驱动里不可能一个相关符号都没有。**

## 六、这个结论的实际意义

### 对「解锁 TC」这个目标

**不可能。**
不是"被锁住了打不开"，是"这个功能在硅片上不存在"。

**要获得 TC，唯一的办法是换卡。**

### 对 MX450 的真实定位

| 项目 | 真实值 |
|---|---|
| FP32 | 2.12 TFLOPS（理论 2.5，达 85%） |
| FP16（模拟） | 0.33–0.60 TFLOPS |
| **Tensor Core** | **无** |
| RT Core | 无 |
| 显存 | 1.68 GB 可用 / 2 GB 物理 |
| 7B QLoRA 需求 | ~4 GB |

**MX450 是一块规格完整的入门级 FP32 显卡，它的 FP32 效率（85%）甚至相当不错。
它只是没有 TC，也不可能有。**

### 对本项目的价值

**这份研究仍然有价值，但价值定位变了：**

不是"我尝试解锁 TC 但失败了"，
而是"**我用九条路径证明了 TU117 没有 TC，并留下了完整的证据链和方法论**"。

**其中包括：**

- 一套可复现的 Turing 逆向工具链（PRAMIN、Falcon IMEM dump、nvidia.ko 符号分析）
- FECS/GPCCS 微码的完整 dump 与反汇编（公开资料未见）
- Falcon IO 地址空间的完整模型（`HOST_IO_INDEX` 仅 6 位）
- nvidia.ko 的 GRCTX 布局（186 个字段全部定位）
- **一条重要的方法论：判断"有没有"比判断"能不能开"更重要，且更早该做**

## 七、方法论教训（写给自己）

**这个项目最大的教训：在投入大量精力寻找"解锁方法"之前，
应该先用最便宜的方法确认"这个东西到底存不存在"。**

**最便宜的方法是什么？**
1. 查公开规格（30 秒）—— 超能网 2019 年就写清楚了
2. 看驱动里有没有相关符号（1 分钟）—— `strings nvidia.ko | grep -i tensor` 零命中
3. 对比同类硬件的性能量级（5 分钟）

**这三步加起来不到 10 分钟，能省掉整个项目的 20 小时。**

**而我们是在 20 小时后，才通过一次联网检索发现的。**

**这就是"每步都联网搜索"的真正价值——它不只是补充信息，
它能防止你在一个错误的前提上建造一整座城堡。**

## 八、最终数据汇总

```
设备: NVIDIA GeForce MX450 (TU117)
SM: 14    capability: 7.5    显存: 2048 MiB (1681 MiB 可用)
驱动: 580.142    CUDA: 13.0

FP32 峰值:  2.122 TFLOPS  (volta_sgemm_128x64_nn)
FP16 峰值:  0.327 TFLOPS  (CUDA core)
mma.sync:   0.600 TFLOPS  (手写 PTX)
cuBLAS TC:  0.320 TFLOPS  (cutlass_70_tensorop_f16_s884gemm)

Tensor Core: 不存在（TU117 移除，位置为 FP16 单元）
```

## 九、致谢与声明

**本结论基于：**
- 3 篇独立公开资料（超能网 / Notebookcheck / 百度百科）
- 4 组本机实测（FP32 / FP16 / mma.sync / cuBLAS TENSOR_OP）
- 1 次 cuBLAS kernel 名抓取（`cutlass_70_tensorop_f16_s884gemm`）
- 9 条逆向路径的排除性证据

**所有实测数据可用 `tools/bench_fp16_vs_tc.py` 与 `tools/profile_kernel2.py` 复现。**

**此前 findings 中所有「TC 被限速」的表述，以本文为准。**
