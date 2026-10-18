# MX450 (TU117) Tensor Core 逆向研究

**简体中文** | [繁體中文](README.zh-TW.md) | [English](README.en.md)

> 查清 NVIDIA GeForce MX450（TU117M）上 Tensor Core 被降速至理论值 5–10% 的机制，
> 并系统性地探明这台机器在固件/驱动栈上的所有可访问边界。
>
> **状态：公开资料可达的边界已全部探明 · 未实现解锁 · 结论以实测数据为准**

---

## 一句话结论

**这块卡上确实存在一个被锁住的算力空间（Tensor Core 有指令、能执行、但吞吐只有理论值的 5–10%）。
这台机器用五层机制把它封住了，本次研究逐一验证并绕过了其中四层——第五层是 NVIDIA 从未公开的寄存器语义。**

---

## 1. 项目起点：一个反常的实测现象

| 路径 | 实测吞吐 | 说明 |
|---|---|---|
| 手写 PTX `mma.m16n8k8`（Tensor Core 指令） | **0.60 TFLOPS** | 走 TC 路径 |
| 手写 PTX `FP16x2`（CUDA 核，不碰 TC） | **5.77 TFLOPS** | 纯 FP16 单元 |
| 手写 PTX `FP32 FMA` | 3.06 TFLOPS | 基准 |
| `cuBLAS` FP16，日志确认 `CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)` | **0.32 TFLOPS** | NVIDIA 自己的库 |
| `cuBLAS` FP32（不用 TC） | 2.07 TFLOPS | 对照 |
| TU117 理论张量峰值（14 SM × 4 TC） | 6 – 12 TFLOPS | — |

**TC 指令存在、能编译、能执行、不报错 —— 但比不用 TC 还慢 5–6 倍。**

---

## 2. 五层封锁与突破结果

| # | 封锁层 | 常规认知 | 本次结果 |
|---|---|---|---|
| ① | **硬件层**：GPU 无独立 SPI Flash | 无法刷 vBIOS | ❌ 无法突破（物理事实） |
| ② | **系统固件**：Intel Boot Guard 签名加密 | 无法解析 BIOS | ❌ 无法突破（需厂商私钥） |
| ③ | **GPU 固件**：GSP FWSEC 证书链验证 | 无法修改微码 | ❌ 无法突破（需 NVIDIA 私钥） |
| ④ | **驱动层**：NVML / CUPTI / 私有 ioctl 逐层裁剪 | 无法读取底层 | ✅ **部分突破**（PRAMIN 通道） |
| ⑤ | **寄存器语义**：GPC/TPC 偏移未公开 | — | ⚠️ **未解**（信息不存在） |

---

## 3. 已证实的 21 条硬事实

| # | 事实 | 证据 |
|---|---|---|
| 1 | TC 存在但被限速至理论值 5% | 手写 PTX 0.60 vs 理论 6–12；cuBLAS 同现象 |
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
| 18 | **该段语义未知** | 三条验证路线均失败 |
| 19 | PGRAPH 在 `0x400000` | envytools rnndb 权威定义 |
| 20 | 微码不可修改 | FWSEC 签名验证 |
| 21 | 显存 1.68 GB 不可变 | 物理焊死 |

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
