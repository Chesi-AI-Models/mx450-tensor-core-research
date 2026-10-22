# D1 GEMM 优化 + D2 锁频实验 —— 完整实测报告

**简体中文** | [繁體中文](D1-GEMM优化与D2锁频实测报告.zh-TW.md) | [English](D1-GEMM优化与D2锁频实测报告.en.md)

日期：2026-10-20
GPU：NVIDIA GeForce MX450 (TU117)，14 SM，CC 7.5，VRAM 2GB
驱动：nvidia-open 580.142（GSP 强制启用）
测试规模：N=2048，FP16 输入 / FP32 累加，FLOP = 2·N³ = 1.718e10

---

## 一、结论摘要

| 指标 | 结果 |
|---|---|
| 自研 FFMA GEMM 最好成绩 | **2.4872 TFLOPS**（C7 `64×128×BK8 T8×8 PAD0 单缓冲`） |
| 相对 cuBLAS fp32 mm（同进程测） | **1.17x**（cuBLAS 2.13 TFLOPS） |
| 相对 cuBLAS fp16 mm | **8.1x**（cuBLAS fp16 仅 0.32 TFLOPS） |
| 时钟折算效率 | **80–81%**（对比：纯 FP32 FFMA 可到 99.0%） |
| `nvidia-smi -lgc` 锁频 | **被接受（rc=0）但完全无效** |
| 热降频标志 | **Not Active**（不是散热压的） |

**★ 最重要的修正：「3.06 TFLOPS」这个分母是错的。**
它来自 1665 MHz 假设，但本机 SM 时钟实测在 **1290–1830 MHz** 之间漂移
（`clocks.max.sm = 2100`）。因此：

- 物理峰值 = `2 × 64 × 14 × clk` = 1290MHz→2.31 / 1395→2.50 / 1665→2.98 / 1830→3.28 / 2100→3.76 TFLOPS
- **绝对 TFLOPS 跨轮次不可比**，只有「相对同轮时钟折算峰值」的百分比可比
- 教训与 T2-1 的「MB/GB 单位混用」同类：**分母没校准，所有百分比都是假的**

---

## 二、D1 优化过程（每一步都是实测数据）

| 版本 | 改动 | TFLOPS | 时钟折算效率 | 结论 |
|---|---|---|---|---|
| v2 | 标量 smem，BK=8 PAD=6，128×128×256 | 1.603 | 66.3% | 基线 |
| v4 | A tile 转置 + BK×PAD 扫描 | 1.876 | 77.6% | 转置方案证伪（16-bit 写冲突 > 读收益），但 BK/PAD 扫描有效 |
| v5 | **float4/uint4 全局加载**（16B） | 1.9714 | 81.6% | **有效**；最优 A:v4 B:v4 BK=16 PAD=8 |
| v6 | float smem + A 转置 + 内层 LDS.128 | 1.811 | — | **倒退**：smem 字节翻倍 + 转置标量写入冲突 |
| v7/v8 | tile × 线程 × 双缓冲扫描 | 2.3075→2.4872 | ~81% | **双缓冲有效**；BK=8 优于 BK=16 |
| C 轮（校准） | 独立进程 + cuBLAS 校准点 | **2.4872** | 80.9% | 见下表 |

### 2.1 校准后的配置对比（两轮独立扫描，数字可复现）

| 配置 | best TFLOPS（轮1 / 轮2） | regs | blocks/SM |
|---|---|---|---|
| C1 128×128×8 P8 单 | 2.2157 / 2.1472 | 122 | 2 |
| C2 128×128×8 P8 **双** | 2.4407 / 2.4578 | 128 | 2 |
| C3 128×128×16 P8 单 | 2.3527 / 2.2902 | 124 | 2 |
| C4 128×128×16 P8 双 | 2.3977 / 2.3295 | 125 | 2 |
| C5 128×64×16 P8 单 | 2.4333 / 2.3953 | 125 | 4 |
| C6 128×64×16 P8 双 | 2.3459 / 2.2580 | 127 | 4 |
| **C7 64×128×8 P0 单** | **2.4872 / 2.4755** | 122 | 4 |
| **C8 128×128×8 P16 双** | 2.4851 / 2.4766 | 128 | 2 |
| 校准点 cuBLAS fp32 | 2.1333 / 2.1471 | — | — |
| 校准点 cuBLAS fp16 | 0.3191 / 0.3171 | — | — |

同轮漂移仅 1.02–1.09x，说明**独立进程 + 交错 + cuBLAS 校准**这套方法是可靠的。

### 2.2 关键机制验证

**① float2 ≠ 2 个 half（踩坑）**
CUDA 的 `float2` 是 2 个 **float** = 8 字节。要表示 2 个 half（4 字节）必须用
`unsigned int`。用错立刻触发 `misaligned address`，而这是**粘性错误**，会打死整个
context，导致后续所有 kernel 报 716/700。→ 必须写**静态对齐预检**，不能靠跑崩来验证。

**② padding 会破坏 16B 对齐（联网查证 + 实测一致）**
A tile `[BM][BK+PAD]`，要 float4（16B）写共享内存，需 `(BK+PAD)×2 % 16 == 0`，
即 `BK+PAD` 必须是 8 的倍数。BK=8 时 PAD=6 → 行首偏移 12B，静态预检直接拦下。
可选 PAD ∈ {0, 8, 16}。**PAD=0 时行跨度正好 16B，天然零 bank 冲突**（实测 C7 用 PAD=0）。

**③ 减少内层指令数不等于变快**
v6 把内层从 `16×LDS.U16 + 16×cvt` 改成 `4×LDS.128`，指令数降到 1/4，
但**整体慢了 8%**。原因是 smem 存 float 让字节量翻倍、且转置写入是 8 次标量 STS。
→ 「指令数」不是唯一维度，**smem 字节流量**同样关键。

**④ 纯 FFMA 天花板可信（99.0%）**
`2.4747 TFLOPS @ 1395 MHz`，分母 `2×64×14×1.395 = 2.4998` → 99.0%。
说明公式与 FMA 管线都能跑满，GEMM 的 80.9% 差的是**访存停顿**，不是算力。

---

## 三、D2 锁频 / 超频实验（全部实机，root）

| 手段 | 命令 | 结果 |
|---|---|---|
| 锁 SM 时钟 1500 | `nvidia-smi -lgc 1500` | rc=0「GPU clocks set」，**实测仍 1320** |
| 锁 1650 / 1800 / 1950 / 2100 | 同上 | rc=0，实测 1305 / 1290 / 1290 / 1290，**GEMM 反而降到 1.85** |
| 功耗上限 | `nvidia-smi -pl 35` | **Changing power management limit is not supported** |
| 热降频标志 | `clocks_event_reasons.active` | **0x0000000000000000**，hw/sw thermal slowdown 全 Not Active |
| 功耗读数 | `power.draw` / `power.limit` | **[N/A] / [N/A]**（default/max 报 5001 W，无效值） |
| 风扇转速 | `fan.speed` + `/sys/class/drm/card*/device/hwmon/*/pwm*` | **[N/A]，无 pwm 节点** |

### 3.1 为什么 -lgc 无效（推断 + 已有证据支撑）

- 命令被驱动**接受并回显设置成功**，但时钟纹丝不动 → 不是权限问题、不是语法问题
- **热降频标志全是 0** → 不是散热压的（温度 80–86°C，但没触发 throttle flag）
- 本项目已实测确认：**开源模块强制启用 GSP**（`NVreg_EnableGpuFirmware=0` 被忽略）
- GSP-RM 固件接管了时钟/电源域 → `nvidia-smi` 的 lgc 走的是老的 PMU 路径，被旁路

**→ 想真正超频，只有两条路：**
1. **换闭源模块** + `NVreg_EnableGpuFirmware=0`，让时钟管理回到驱动侧（open-gpu-kernel-modules
   issue #965 有成功案例，代价是某些场景性能降 50%）—— 尚未做
2. 改 vBIOS/PPTable —— **本机无 vBIOS**（ROM BAR 分配失败，已实测），此路不通

---

## 四、文件清单

| 文件 | 作用 |
|---|---|
| `sw_tensor_core_v5_vec.py` | float2/float4 全局加载 + 静态对齐预检 |
| `sw_tensor_core_v6_fp32smem.py` | float smem + A 转置 + 内层 LDS.128（证伪） |
| `sw_tensor_core_v7_sweep.py` | tile × 线程 × 双缓冲首轮扫描 |
| `sw_tensor_core_v8_around_s5.py` | S5 周边细扫（PAD/BK/tile） |
| `d1_diag_ceiling.py` | 寄存器 / 占用率诊断 |
| `d1_bench_calibrated.py` | **独立进程 + cuBLAS 校准点**（推荐基准工具） |
| `d1_peak_truth.py` | 纯 FP32/FP64 FMA 峰值 + 时钟反解 |
| `d1_clock_during.py` | 内核运行期间时钟采样 |
| `d2_clock_lock.py` | 锁频/功耗上限实验（需 root） |

---

## 五、下一步

1. **D1 剩余 8–18%**：目标把时钟折算效率从 81% 提到 ≥90%（对齐 cuBLAS fp32 的 89.6%）。
   方向：3 级流水（提前两块）、减少 `__syncthreads` 次数、拆 cvt 到 load 阶段
2. **D2 唯一可行路**：换闭源模块验证 `-lgc` 是否生效（可逆，需记录恢复步骤）
3. 所有后续基准**必须**用 `d1_bench_calibrated.py` 的方法（独立进程 + 同轮 cuBLAS 校准）
