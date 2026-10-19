# MX450 (TU117) Tensor Core 逆向研究

**简体中文** | [繁體中文](README.zh-TW.md) | [English](README.en.md)

> 一个关于「**这块卡到底有没有 Tensor Core**」的完整取证过程。
> 结论：**没有。** 本文记录如何用九条路径证明这一点，以及沿途挖出的
> Turing 固件、Falcon 微码、GR Context 布局等一手材料。
>
> **状态：结论已闭合 · 全部实测可复现**

---

## 一句话结论

**TU117 的 SM 里没有 Tensor Core。原本放 TC 的位置是 FP16 单元。**

**HMMA 指令能编译、能执行、结果正确——因为它由 FP16 单元模拟，而不是由 TC 执行。**

**实测：FP16 打包运算 5.77 TFLOPS，而 HMMA 指令只有 0.60 TFLOPS——TC 路径比纯 FP16 慢 9.6 倍。**

**所以「解锁 TC」不成立：不是被锁住打不开，是这个功能在硅片上不存在。**

---

## 0. 为什么这个结论可信：三方公开资料 + 四组实测

### 公开资料（三方独立，2019 年就已写明）

| 来源 | 原文 |
|---|---|
| **超能网** 2019-04-24 | 「GTX 1650 所用的 TU117 和之前的 TU116 一样**去掉了 RT Core、Tensor Core**，SM 单元内部架构经过了修改，**原本放 Tensor Core 的位置变成了 FP16 半精度浮点运算单元**」 |
| **Notebookcheck** | 「与更快的 RTX 2000 GPU 相比，**1650 没有集成 Raytracing 或 Tensor 内核**」 |
| **百度百科** | 「TU117 **移除了 RT Core 和用于 AI 超采样的 Tensor Core**」 |
| 对照：RTX 20 系列 | 每个 SM 有 **8 个 Tensor Core** |

### 本机实测（`tools/` 下脚本可复现）

| 路径 | 实测 | 说明 |
|---|---|---|
| PTX `FP16x2`（打包，CUDA 核） | **5.77 TFLOPS** | 纯 FP16 单元 |
| PTX `FP32 FMA` | 3.06 TFLOPS | 基准 |
| cuBLAS FP32（`volta_sgemm_128x64_nn`） | 2.12 TFLOPS | 达理论 85% |
| **PTX `mma.m16n8k8`（TC 指令）** | **0.60 TFLOPS** | **比 FP16 慢 9.6×** |
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)` | 0.32 TFLOPS | 比 FP16 慢 18× |

**FP16 单元理论峰值核算：`14 SM × 64 CUDA × 2(FMA) × 2(打包) × 1.6 GHz = 5.73 TFLOPS`**
**实测 5.77 —— 吻合。说明 FP16 单元是满速的。**

**而 TC 路径 0.60，是 FP16 的 1/9.6。**

### 决定性证据：kernel 名

cuBLAS 日志显示它选了 TC 算法：
```
algo: val=CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)
```

但实际执行的 kernel 是：
```
void cutlass::Kernel<cutlass_70_tensorop_f16_s884gemm_relu_f16_1...
                    ↑ cutlass_70（SM 7.0 编译）  ↑ s884 = m8n8k4（Volta 第一代 TC 形状）
```

**两个破绽：**

1. **`cutlass_70` 而不是 `sm75`** —— NVIDIA **没有为 TU117 提供 sm75 的 TC kernel**，cuBLAS 只能回退到 Volta 的通用 kernel
2. **`s884`（m8n8k4）** —— 不是 Turing 的 m16n8k8 / m16n8k16

**如果 TU117 真有 TC，NVIDIA 不可能不为它编一个 sm75 的 kernel。**

---

## 1. 项目起点：一个被误读的现象

**最初看到的是这个：**

| 路径 | 实测吞吐 |
|---|---|
| 手写 PTX `mma.m16n8k8`（TC 指令） | 0.60 TFLOPS |
| 手写 PTX `FP16x2`（CUDA 核，不碰 TC） | **5.77 TFLOPS** |
| 手写 PTX `FP32 FMA` | 3.06 TFLOPS |
| `cuBLAS` FP16 `TENSOR_OP` | 0.32 TFLOPS |
| `cuBLAS` FP32 | 2.07 TFLOPS |

**当时的解读是：「TC 存在，但被限速到理论值（6–12 TFLOPS）的 5–10%」。**

**这个解读错了两处：**

1. **「6–12 TFLOPS 理论值」是拿 RTX 20 系列（TU10x，每 SM 8 个 TC）套过来的，
   TU117 根本没有 TC，这个基准不存在。**
2. **「比不用 TC 还慢 5–6 倍」这个现象，本身就是「模拟」的典型特征——
   真 TC 不可能比同代 FP16 慢，只有「用别的单元模拟 TC」才会这样。**

**正确解读：`mma` 慢于 `FP16x2`，说明 HMMA 指令被降级到 FP16 单元执行。**

---

## 2. 九条路径的探索结果

**在得出结论之前，本项目把软件层能走的路径全部走了一遍。**
**这些路径本身是有价值的——它们共同构成了「TU117 上没有可解锁的 TC」的排除性证据。**

| # | 路径 | 做了什么 | 结果 |
|---|---|---|---|
| 1 | 用户态 API | NVML / CUPTI / 私有 ioctl | 逐层裁剪，CUPTI `rc=38` 零 metric |
| 2 | 系统固件 | flashrom 只读导出 24 MB BIOS | Intel Boot Guard 加密，无 VFCT / 无 `_ROM` |
| 3 | GPU 固件 | GSP 双层 RISC-V ELF 完整反汇编 | 30 MB 全解，8 个 Ampere 寄存器地址零命中 |
| 4 | 显存访问 | PRAMIN 通道（BAR0+0x700000） | **打通**，写-读回验证，dump 32 MB |
| 5 | Falcon 安全模式 | 读 `HWCFG1` | **LIGHT SECURE**（非 HS 黑箱） |
| 6 | FECS 微码 | GPU 满载时读 IMEM | **完整 dump 32 KB + 反汇编 11401 行** |
| 7 | GPCCS 微码 | 同上，基址 0x41A000 | **完整 dump 16 KB + 反汇编 6059 行** |
| 8 | Falcon IO 空间 | 查 `HOST_IO_INDEX` 位宽 | 256 KB 空间，host 窗口仅 4 KB，`0x20000` 段不可见 |
| 9 | 闭源驱动 | nvidia.ko 符号级逆向（46614 符号） | 五类线索全部排除，**`tensor|HMMA|MMA_` 字符串零命中** |

**九条路径，没有一条找到「TC 配置」或「算力开关」。**

**原因现在很清楚了：因为它们要找的东西不存在。**

---

## 3. 已证实的硬事实

| # | 事实 | 证据 |
|---|---|---|
| 1 | **TU117 无 Tensor Core；HMMA 由 FP16 单元模拟** | 三方公开资料 + FP16 5.77 vs mma 0.60（慢 9.6×）+ kernel 名 `cutlass_70` |
| 2 | GSP 固件不含 TC 控制逻辑 | 双层 ELF 反汇编，8 个 Ampere 寄存器地址零命中 |
| 3 | vBIOS 受 FWSEC 证书链保护 | 固件内 8 个 `VERIFY_*` 错误码 |
| 4 | BAR1「仅映射 16MB」可绕过 | PRAMIN 通道，写-读回验证通过 |
| 5 | P2P「消费级 EINVAL」可绕过 | 不走 P2P，直走 PRAMIN |
| 6 | 驱动运行时 BAR0 可访问 | 内核模块 `ioremap` |
| 7 | 显存可读 | 32 MB dump，MD5 可校验 |
| 8 | 显存不含 Falcon 微码 | Kepler 微码签名零命中 |
| 9 | FECS Falcon 基址 = `0x409000` | 寄存器实测 |
| 10 | `HWCFG1` 位于 `+0x12c`（非 `+0x10c`） | 源码确认 |
| 11 | **FECS 运行在 LIGHT SECURE 模式** | `security_model = 2`（非 HS 黑箱） |
| 12 | **FECS 微码可完整读取** | 32 KB 全量落盘 |
| 13 | 微码与 gk20a 同源结构 | 7 字节签名 `00fe0400bda47e` 匹配 |
| 14 | 入口 = 设置栈顶 + `lcall main` + `exit` | 反汇编确认 |
| 15 | ISA 为 fuc6 | `core_rev = 6` |
| 16 | 449 条寄存器写入，地址 100% 解析 | 静态分析 |
| 17 | 379 次落在 `0x20000–0x2FFFF` 段 | 分段统计 |
| 18 | **该段是 Falcon IO 空间，非 BAR0** | 五种假设实测排除 + 文档确认 |
| 19 | PGRAPH 在 `0x400000` | envytools rnndb 权威定义 |
| 20 | 微码不可修改 | FWSEC 签名验证 |
| 21 | 显存 1.68 GB 不可变 | 物理焊死 |
| 22 | **Falcon IO 映射规则 `Falcon = Host × 0x40`** | 两处独立验证（`0x12c`=UC_CAPS2、`0x184`=CODE） |
| 23 | **GPCCS Falcon 基址 = `0x41A000`** | HWCFG 签名扫描 |
| 24 | **GPCCS 微码可完整读取（16 KB）** | MD5 `784c08e5...`，7 字节签名匹配 |
| 25 | **FECS/GPCCS 地址集是包含关系** | GPCCS 的 38 个地址全在 FECS 的 72 个内 |
| 26 | **两者通过 `0x246FC/0x247FC/0x24FFC` 通信** | 三地址访问次数完全相同；微码自算 `0x23300+0x14FC=0x247FC` |
| 27 | **`iowr` 有两套地址机制** | `bit31=0`→Falcon IO；`bit31=1`→BAR0 绝对地址 |
| 28 | **Falcon IO 空间 256 KB，host 窗口 4 KB** | envytools 文档 + `HOST_IO_INDEX` 位宽 |

---

## 4. 关键突破：如何读出 FECS 微码

五道锁逐一打开：

| 锁 | 常规认知 | 绕过方法 |
|---|---|---|
| BAR1 只映射 16 MB | 读不到显存 | 改走 **PRAMIN**（`BAR0+0x700000`） |
| P2P 对 GeForce 返回 EINVAL | 无法 pin VRAM | 不走 P2P |
| 驱动运行时 BAR0 被占用 | 无法访问寄存器 | **内核模块 `ioremap`** |
| Falcon 若 HS 则黑箱 | 完全不可读 | **实测为 LS**（关键） |
| 空闲时 Falcon `halted=1` | IMEM 全零 | **制造 GPU 满载 100%** |

**读取流程**：

```c
// 1. GPU 满载（torch 跑 GEMM 循环）
// 2. insmod mx450_p4.ko bar0_phys=<BAR0> mode=dump dump_bytes=32768
// 3. 模块内部：
//    ioremap(BAR0 + 0x409000, 0x1000)
//    读 HWCFG1 确认 security_model == 2 (LS)
//    保存 IMEMC(0) 原值
//    for (addr = 0; addr < 32768; addr += 4) {
//        blk  = addr / 256;
//        word = (addr % 256) / 4;
//        writel((blk<<8)|(word<<2), base + 0x180);   // IMEMC
//        *(u32*)(buf+addr) = readl(base + 0x184);    // IMEMD
//    }
//    恢复 IMEMC(0)
```

---

## 5. 交付物

| 产物 | 大小 | 校验 |
|---|---|---|
| `fecs_imem.bin` — FECS 微码原始镜像 | 32 KB | MD5 `ea8eb3c46165b00421896eef45d1070a` |
| `fecs_plain.asm` — 完整反汇编 | 1.3 MB / 11401 行 | — |
| `vram_dump.bin` — 显存 dump | 32 MB | MD5 `1b224dcfeec507673ee64fd35d44005c` |
| `bios_readonly.bin` — 系统 BIOS | 24 MB | MD5 `4d2c99e85e761e0a3b1b8dcada4cc950` |
| `envytools/envydis` — 反汇编器 | — | 自行编译 |

> 二进制产物因体积未纳入仓库，可通过 `tools/` 下的脚本重新生成。

---

## 6. 仓库结构

```
.
├── README.md                    简体中文（本文件）
├── README.zh-TW.md              繁體中文
├── README.en.md                 English
├── LICENSE
├── docs/
│   ├── 02-技术路线图.md
│   └── 03-内核模块路线图.md
├── findings/                    二十三篇实测记录（append-only）
├── refs/                        NVIDIA 官方头文件（MIT，原样引用）
├── kernel/                      6 个内核模块
│   ├── mx450_probe.c            工具链验证
│   ├── mx450_p1.c               kprobe 符号解析
│   ├── mx450_p2.c               内核内存 no-fault 读取
│   ├── mx450_p3.c               PRAMIN 显存访问
│   └── mx450_p4.c               FECS IMEM 读取
├── tools/                       20 个分析工具
└── scripts/                     QEMU 隔离测试等
```

---

## 7. 测试环境

| 项目 | 值 |
|---|---|
| 机器 | LYLNAS（联想昭阳 E4-ITL 改装） |
| 系统 | fnOS 1.2.0505 / Debian 12 / kernel 6.18.18 |
| 主板 / BIOS | LNVNB161216 / F8CN59WW(V2.22) 2024-06-14 |
| CPU | Intel i7-1165G7（4C8T） |
| GPU | GeForce MX450，TU117M，`10de:1f97`，`NV_PMC_BOOT_0 = 0x167000a1` |
| 显存 | 2048 MiB GDDR6，64-bit，单次分配上限 1620 MiB |
| 驱动 | 580.142（闭源）/ CUDA 13.0 |
| 工具链 | gcc 12.2 / g++ 12.2 / cmake 3.25 / envytools / flashrom / strace |

---

## 8. 复现

```bash
# 编译内核模块
cd kernel && make

# 读取 FECS 微码（需 GPU 满载）
insmod mx450_p4.ko bar0_phys=<BAR0> mode=dump dump_bytes=32768

# 反汇编
envydis -m falcon -V fuc6 -i -n fecs_imem.bin > fecs_plain.asm

# 分析寄存器访问
python3 tools/analyze_iowr.py
```

---

## 9. 未解的边界

**`0x20000` 段的语义无法确定**，三条路线全部尝试：

| 路线 | 结果 |
|---|---|
| envytools rnndb | 覆盖止于 Maxwell，且上界 `0x227FF` 小于微码实际访问的 `0x23800` |
| NVIDIA 开源头文件 | `turing/tu102/` 无 `dev_gpc.h` / `dev_tpc.h` |
| NVIDIA 开源 HAL | `NV_PGRAPH_GPC` / `NV_PGRAPH_TPC` 零定义，逻辑在闭源 `nv-kernel.o_binary` |

**这不是"还没找到"，是"这一层信息 NVIDIA 没有公开"。**

---

## 10. 风险声明

1. **可能损坏硬件。** 向 GPU 寄存器写入未公开值可导致无法初始化、黑屏。
2. **笔记本独显无独立 vBIOS Flash。** 写错配置后**没有可回滚的固件区**。
3. **不提供任何成功保证。** 本项目已确认：微码受 FWSEC 签名保护，只能看不能改。
4. **显存 1.68 GB 是物理焊接**，任何软件手段都无法改变。
5. **仅供研究与教育用途**，仅适用于你自己合法拥有的硬件。

---

## 11. 声明

本项目与 NVIDIA 无任何关联。所有商标归其各自所有者。
仓库内不含 NVIDIA 专有代码或固件；`refs/` 下均为 NVIDIA 以 MIT 许可公开发布的原始头文件。

## 许可

MIT（见 `LICENSE`）。研究成果欢迎引用，使用风险自负。
