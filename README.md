# MX450 (TU117) Tensor Core 逆向研究

> 查清 NVIDIA GeForce MX450（TU117M）上 Tensor Core 被降速至理论值 5–10% 的机制，
> 并系统性地探明这台机器在固件/驱动栈上的所有可访问边界。
>
> **状态：三条高层路径已全部探底 · 未实现解锁 · 结论以实测数据为准**

---

## 一句话结论

**这块卡上确实存在一个被锁住的算力空间（Tensor Core 有指令、能执行、但吞吐只有理论值的 5–10%），
而这台机器用三层机制把这个空间封住了：GPU 无独立固件、系统 BIOS 受 Boot Guard 加密、
用户态驱动 API 逐层裁剪。**

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

这不是「没有 Tensor Core」，而是「有，且被压到几乎不可用」。

## 2. 四条互证证据

| # | 证据 | 方法 |
|---|---|---|
| 1 | 生成的是**真张量指令**，不是 FMA 展开 | `ptxas -arch=sm_75` 编译后对比 `.text`：mma 版与 ffma 版同为 896 B / 56 条，但 opcode 指纹截然不同（ffma 版 `2595×32`；mma 版 `572×8 + 2375×6 + 2328×6…`） |
| 2 | 硬件**能执行**该指令 | 运行不抛非法指令，结果正确 |
| 3 | 吞吐只有理论值 5–10%，且**低于纯 FFMA** | 8 路独立累加器 + 8 warp + 2048 blocks 压测，三次运行恒为 0.60 TFLOPS，零抖动 |
| 4 | **NVIDIA 自家 cuBLAS 同样如此** | `CUBLAS_LOGINFO_DBG` 显示选择 `TENSOR_OP(99)`，实测 0.32 TFLOPS，比 FP32 慢 6.4 倍 |

**旁证**：`NV_PMC_BOOT_0 = 0x167000a1` → chipset `0x167` = TU117（读取方式见 `tools/regs_peek.sh`）。

## 3. 三条路径的探底结果

| 路线 | 手段 | 结果 | 卡点性质 |
|---|---|---|---|
| **A** | CUPTI metric 枚举 | `cuptiDeviceEnumMetrics -> rc=38`，0 个 metric；写入 `NVreg_RestrictProfilingToAdminUsers=0` 并重载驱动后**仍为 rc=38** | 库/平台限制，**已排除权限因素** |
| **B** | `NV2080_CTRL_GR_GET_INFO` 私有 ioctl | 通道打通（`hClient=0xc1d00063`），`DEVICE_0/SUBDEVICE_0` 返回占位句柄 `0xcaf0xxxx`，`GR_GET_INFO` 全部 `status=31` | 驱动对象模型隔离 |
| **C** | 系统 BIOS dump → 提取 vBIOS | flashrom 只读导出 24 MB；**无 `VFCT` 表、无 PCIR/`NVIDIA`/`TU117` 字符串、无 LZMA/Tiano 压缩头**；`0x800000` 起为密文 | **Intel Boot Guard 签名加密** |

详细过程见 `findings/`。

## 4. 这台机器的四层防线

```
┌─ ① 硬件层 ───────────────────────────────────────────┐
│ GPU 无独立 SPI Flash                                  │
│   卸载驱动后 PCI ROM BAR 返回 EIO（0 字节）            │
├─ ② 固件层（系统）───────────────────────────────────┤
│ Intel Boot Guard：BIOS 关键区域签名 + 加密            │
│   24 MB dump 中无 VFCT 表、无 _ROM 方法、无明文 PCIR   │
├─ ③ 固件层（GPU）────────────────────────────────────┤
│ GSP FWSECLIC：vBIOS 证书链 + 镜像签名 + 哈希树验证    │
│   → 固件内实测到 VERIFY_CERT_VERIFY_FAIL /           │
│     VERIFY_BIOS_SIG_FAIL / VERIFY_HAT_FAIL 等错误码   │
│   → 改一个字节即被拒绝加载                            │
├─ ④ 驱动层 ───────────────────────────────────────────┤
│ NVML 不暴露 TC 计数                                   │
│ CUPTI metric 枚举 rc=38                               │
│ 私有 ioctl 对象级隔离（占位句柄 0xcaf0xxxx）          │
└──────────────────────────────────────────────────────┘
```

**四层，四种不同的技术手段，每一层单独就足以封死一条路径。**

| 要改的东西 | 拦路的那一层 | 可否绕过 |
|---|---|---|
| vBIOS 镜像（devinit / 限速配置） | ③ GSP 证书验证 | 仅可能通过加载器时序漏洞（Falcon 类），需自研 |
| 系统 BIOS 中的 VGA 模块 | ② Boot Guard | 需要 Intel/联想私钥，**不可能** |
| 运行时寄存器（SS/PLM 类） | ④ 驱动对象模型 + ③ PLM 权限 | 需先知道 Turing 的寄存器地址（**未知**） |
| 显存容量 | 硬件（颗粒焊死） | **不可能** |

## 5. 参考项目与本项目的定位差异

| 项目 | 它做到了什么 | 为什么本机不适用 |
|---|---|---|
| [`cmpunlocker`](https://github.com/amoghmunikote/cmpunlocker) | CMP 170HX（GA100）解锁 64–80 GB HBM2e、恢复 SM 吞吐。原理：利用 Falcon BootROM 的 `.fwsignature_ga100` 加载缺陷，在 GSP 启动前用 ROP 链打开 4 个 PLM 寄存器，再写 `CFG1`/`LMR`/`SS0`/`SS1` | ① GA100 的 HBM 颗粒**物理存在**，MX450 的 2 GB 是焊死的上限；② `FBPA_CFG1`/`PLM`/`SS0`/`SS1` 的地址在 Turing 开源头文件中**完全不存在**（`dev_fbpa.h` 仅 29 行、`dev_fuse.h` 仅 35 行、`gh search PLM` 零命中） |
| [`envytools`](https://github.com/envytools/envytools) | GPU 寄存器数据库、ISA 反汇编、BAR0 peek/poke | rnndb 寄存器库只覆盖 Maxwell 及更早；Turing 无定义 |
| `nouveau` / `NVK` | Turing 命令流与 ISA 的部分逆向 | 不涉及算力解锁 |
| [`NVIDIA/open-gpu-kernel-modules`](https://github.com/NVIDIA/open-gpu-kernel-modules) | Turing+ 内核模块源码（GPL/MIT） | 是**本项目唯一有价值的资料来源**，但关键寄存器未公开 |

**本项目的真正产出**：不是「破解」，而是一张**标好断点的地图** ——
后来者不必再重复这 6 个实验、3 个 bug、3 次走不通。

## 6. 仓库结构

```
.
├── README.md
├── LICENSE                         MIT + 独立研究声明
├── docs/
│   └── 02-技术路线图.md            四阶段计划、判定标准、已知硬约束
├── findings/                       逐日实测记录（append-only，不修改历史）
│   ├── 2026-10-18-阶段1-寄存器调查.md
│   ├── 2026-10-18-阶段1-补记-TC计数查询.md
│   ├── 2026-10-18-阶段1-补记2-ioctl通道.md
│   ├── 2026-10-18-阶段1-补记4-BIOS分析.md
│   └── 2026-10-18-阶段1-补记5-GSP-vBIOS证书验证.md
├── refs/                           NVIDIA 官方头文件（MIT 许可，原样引用）
│   ├── dev_fuse.h  dev_fbpa.h  dev_gsp.h
│   ├── cl0080.h  nvos.h  nv_escape.h  nvstatus.h
│   └── nvidia-3d-init.c            GR_GET_INFO 官方调用参考
├── tools/
│   ├── regs_peek.sh                mmap BAR0 读 PMC 寄存器区
│   ├── sass_probe.py               ptxas 编译 + SASS 机器码比对
│   ├── tc_bench.py                 8 路并行 TC 吞吐压测
│   ├── cublas_kernel.py            抓 cuBLAS 实际算法选择
│   ├── vram_wall.py                显存分配边界 + UVM 超配代价
│   ├── read_tc_count.c             NV2080 ioctl 读 TC 计数（半成品，含 3 个 bug 的修法）
│   ├── enum_cupti_metrics.py       CUPTI metric 枚举（rc=38，已判定不可用）
│   ├── find_vbios.py               BIOS 中定位 vBIOS
│   └── bios_structure.py           BIOS 固件卷结构分析
├── patches/                        驱动补丁（尚未开始）
└── scripts/                        一键复现
```

## 7. 测试环境

| 项目 | 值 |
|---|---|
| 机器 | LYLNAS（联想昭阳 E4-ITL 改装） |
| 系统 | fnOS 1.2.0505 / Debian 12 bookworm / kernel 6.18.18 |
| 主板 / BIOS | LNVNB161216 / F8CN59WW(V2.22) 2024-06-14 |
| CPU | Intel i7-1165G7（4C8T，AVX-512 VNNI） |
| 内存 | 2×8 GB DDR4-3200 双通道（已插满） |
| GPU | GeForce MX450，TU117M，`10de:1f97`，`NV_PMC_BOOT_0 = 0x167000a1` |
| 显存 | 2048 MiB GDDR6，64-bit，实测带宽 69.3 GB/s，**单次分配上限 1620 MiB** |
| 驱动 | 580.142（闭源）/ CUDA 13.0 |
| 工具链 | torch 2.3.1+cu121、ptxas 12.9、cuBLAS 12.0、gcc 12.2、flashrom、strace |

## 8. 复现

```bash
# 显存硬墙（普通分配 vs UVM 超配）
python3 tools/vram_wall.py

# TC 吞吐压测（8 路独立累加器）
python3 tools/tc_bench.py

# 机器码级验证：TC 指令是否被展开成 FMA
python3 tools/sass_probe.py

# cuBLAS 实际算法选择 + 吞吐
/vol1/ganlin/venv/bin/python tools/cublas_kernel.py

# 读 GPU PMC 寄存器（会临时卸载驱动，脚本内自动恢复）
sudo bash tools/regs_peek.sh

# TC 计数查询（当前卡在 DEVICE_0，见 findings 补记2）
gcc -O2 -o /tmp/read_tc_count tools/read_tc_count.c && sudo /tmp/read_tc_count
```

## 9. 已知硬约束（不会因任何后续工作改变）

1. **显存 2048 MiB 是物理焊接的 2 颗 GDDR6**，不是寄存器限制。
   `cuMemAlloc(1640 MiB)` 即返回 `err 2`。任何软件手段都变不出显存。
2. **即使 TC 完全解锁（+10~20× 算力），7B QLoRA 仍需 ≈7 GB 显存**，依然装不下。
   本项目产出的是「算力维度的认知与可能的提升」，不是「能训更大模型」。

## 10. 未排除的后续路径

| 路径 | 前提 | 障碍 |
|---|---|---|
| 运行时拦截 GSP 加载 | 编译 `nvidia-open` 内核模块（Turing 自 570 起支持）+ 定位 Falcon 加载点 | **目标寄存器地址未知** —— Turing 开源头文件中无 PLM/CFG1 定义 |
| 从内存抓已解密 vBIOS | BIOS POST 后、驱动加载前 dump 物理内存 | 需自定义内核模块，且 Boot Guard 会清理 |
| 硬件层测量 | 逻辑分析仪观察 TC 时钟域 | 需拆机 + 专业设备 |

## 11. 风险声明

1. **可能损坏硬件。** 向 GPU 寄存器写入未公开值可导致无法初始化、黑屏，甚至需拆机刷 SPI。
2. **笔记本独显无独立 vBIOS Flash。** 写错配置后**没有可回滚的固件区**。
3. **可能违反保修条款。**
4. **不提供任何成功保证。** 若最终确认限制在 OTP eFuse，结论就是「不可解锁」——
   这同样是有价值的结论，会如实记录。
5. **仅供研究与教育用途**，仅适用于你自己合法拥有的硬件。

## 12. 声明

本项目与 NVIDIA 无任何关联。所有商标归其各自所有者。
仓库内不含 NVIDIA 专有代码或固件；`refs/` 下均为 NVIDIA 以 MIT 许可公开发布的原始头文件，
原样引用并完整保留其版权声明。
参考的开源项目各自遵循其原始许可证。

## 13. 进展

- [x] 确认芯片身份（`NV_PMC_BOOT_0 = 0x167000a1` → TU117）
- [x] 证明 TC 指令真实存在（SASS opcode 指纹）
- [x] 量化 TC 限速（0.60 TFLOPS / 理论 6–12，两条独立路径）
- [x] 显存边界（1620 MiB 上限；UVM 4 GB 可分配但带宽损失 21×）
- [x] 排除 CUPTI 路线（`rc=38`，含开放 profiling 权限复测）
- [x] 打通 NV2080 ioctl 通道（ROOT_CLIENT 成功），定位 3 个真实 bug
- [x] 系统 BIOS 只读导出并判定 Boot Guard 加密
- [x] 在 GSP 固件中确认 vBIOS 证书链验证机制（FWSECLIC）
- [ ] 定位 Turing 的 Tensor Core 限速机制（**需要尚未公开的寄存器地址**）
- [ ] 运行时改写（若找到目标寄存器）
- [ ] 稳定性验证与回滚方案

## 许可

MIT（见 `LICENSE`）。研究成果欢迎引用，使用风险自负。
