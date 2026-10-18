# MX450 (TU117) Tensor Core 解锁研究

> 目标：查清 NVIDIA GeForce MX450（TU117M）上 Tensor Core 被降速 5–6 倍的机制，并尝试在合法拥有的硬件上恢复其理论吞吐。
>
> **状态：研究进行中 · 尚未成功解锁 · 所有结论以实测数据为准**

---

## 这是什么

这是一个**硬件逆向工程研究项目**，研究对象是自己合法拥有的笔记本独显
**NVIDIA GeForce MX450（TU117M，PCI `10de:1f97`）**。

项目的起点是一个反常的实测现象：

| 路径 | 实测吞吐 |
|---|---|
| 手写 PTX `mma.m16n8k8`（走 Tensor Core 指令） | **0.60 TFLOPS** |
| 手写 PTX FP16x2（走 CUDA 核，不碰 TC） | **5.77 TFLOPS** |
| 手写 PTX FP32 FMA | 3.06 TFLOPS |
| `cuBLAS` FP16，日志确认选择 `CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)` | **0.32 TFLOPS** |
| `cuBLAS` FP32（不用 TC） | 2.07 TFLOPS |
| TU117 理论张量峰值（14 SM × 4 TC） | 6 – 12 TFLOPS |

**结论：TC 指令存在、能编译、能执行、不报错，但被降速到理论值的 5–10%，甚至比纯 FP32 还慢 5 倍。**

这是“硬件单元被关闭、指令走慢速模拟路径”的典型特征 —— 也就是说，
**这块卡上存在一个被锁住的算力空间，而不是一个空位。**

## 为什么认为有可能解锁

参考项目 [`cmpunlocker`](https://github.com/amoghmunikote/cmpunlocker) 在 CMP 170HX（GA100）
上证明了同类问题的解法：**不刷固件**（固件有签名校验，硬件层面拒绝加载），
而是在驱动初始化的极早期、在限制生效之前，通过 **BAR0 寄存器写入**改写限速配置。

它用来解锁计算的正是两个「throttle」寄存器：

```
SS0 (0x0082381C) = 0x88888888   # FMA / IMLA throttle OFF
SS1 (0x00823820) = 0x00000008   # IMLA4 throttle OFF
```

MX450 的 TC 降速（5–6 倍）与这种「throttle」特征高度吻合，因此本项目按同一方法论推进。

> ⚠️ 但必须提前说明：CMP 170HX 能成功，前提是 **GA100 芯片物理完整、HBM 颗粒真的焊在板上**，
> 限制只在固件/寄存器层。MX450 是否同样如此，**尚未验证** —— 这正是本项目的第一个待办。

## 仓库结构

```
.
├── README.md                  本文件
├── docs/
│   ├── 01-实测基线.md          已完成的全部硬件实测数据
│   ├── 02-技术路线图.md        分阶段计划与判定标准
│   ├── 03-参考项目分析.md      cmpunlocker / envytools / nouveau 的可借鉴点
│   └── 04-寄存器映射.md        TU117 寄存器地址调查（进行中）
├── tools/
│   ├── regs_peek.sh            mmap BAR0 读取 PMC 寄存器区
│   ├── sass_probe.py           用 ptxas 编译并比对 SASS 机器码
│   ├── tc_bench.py             8 路并行 TC 吞吐压测
│   ├── cublas_kernel.py        抓 cuBLAS 实际选择的算法
│   └── vram_wall.py            显存分配边界 + UVM 超配测试
├── findings/
│   ├── 2026-10-18-基线.md      首轮结论
│   └── raw/                    原始输出留档
├── patches/                    驱动补丁（尚未开始）
└── scripts/                    一键复现脚本
```

## 测试环境

| 项目 | 值 |
|---|---|
| 机器 | LYLNAS（联想昭阳 E4-ITL 改装） |
| 系统 | fnOS 1.2.0505 / Debian 12 bookworm / kernel 6.18.18 |
| CPU | Intel i7-1165G7（4C8T，AVX-512） |
| 内存 | 2×8GB DDR4-3200 双通道 |
| GPU | NVIDIA GeForce MX450，TU117M，`10de:1f97`，`NV_PMC_BOOT_0 = 0x167000a1` |
| 显存 | 2048 MiB GDDR6，64-bit，实测带宽 69.3 GB/s |
| 驱动 | 580.142（闭源）/ CUDA 13.0 |
| 工具链 | torch 2.3.1+cu121、ptxas 12.9、cuBLAS 12.0 |

## 复现

```bash
# 1. 读寄存器（会临时卸载驱动，自动恢复）
sudo bash tools/regs_peek.sh

# 2. 机器码级验证：TC 指令是否被展开
python3 tools/sass_probe.py

# 3. TC 吞吐压测
python3 tools/tc_bench.py

# 4. cuBLAS 算法选择与吞吐
/vol1/ganlin/venv/bin/python tools/cublas_kernel.py

# 5. 显存边界
python3 tools/vram_wall.py
```

## 风险声明（务必阅读）

1. **可能损坏硬件。** 向 GPU 寄存器写入未公开的值可能导致 GPU 无法初始化、黑屏、
   甚至需要拆机刷 SPI 才能恢复。本项目所有写操作都应先在可弃置的硬件上进行。
2. **笔记本独显无独立 vBIOS Flash。** 一旦写入错误配置导致 GPU 无法初始化，
   **没有独立的固件区可回滚**，恢复手段极其有限。
3. **可能违反保修条款。**
4. **本项目不提供任何成功保证。** 如果最终查明限制在 OTP eFuse（物理熔断），
   结论就是「不可解锁」—— 这同样是有价值的结论，会如实记录。
5. **仅供研究与教育用途。** 仅适用于你自己合法拥有的硬件。

## 声明

本项目与 NVIDIA 无关联。所有商标归其各自所有者。
项目内不包含任何 NVIDIA 专有代码或固件，仅通过公开接口与自有硬件交互。
参考的开源项目（cmpunlocker、envytools、nouveau、open-gpu-kernel-modules）各自遵循其原始许可证。

## 进展

- [x] 确认芯片身份（`NV_PMC_BOOT_0 = 0x167000a1` → TU117）
- [x] 证明 TC 指令存在（SASS opcode 指纹）
- [x] 证明 TC 被降速 5–6 倍（手写 PTX + cuBLAS 双路径）
- [x] 显存边界实测（1620 MiB 上限）
- [ ] **判断限速是 eFuse 还是寄存器（关键分叉点）**
- [ ] 定位 TU117 的 SS0/SS1 等效寄存器
- [ ] 尝试运行时改写
- [ ] 如需 PLM 绕过，评估 Falcon BootROM 路径

## 许可

MIT（见 LICENSE）。研究成果欢迎引用，但请自行承担使用风险。
