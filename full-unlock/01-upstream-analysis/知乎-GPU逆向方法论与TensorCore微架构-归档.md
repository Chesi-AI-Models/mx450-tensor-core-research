# 知乎两篇 GPU 逆向方法论文章 —— 摘要归档与对本项目的映射

> 来源（2026-10-20 阅读）：
> ① 《GPU逆向方法总结》 https://zhuanlan.zhihu.com/p/2014778262367250116 （编辑于 2026-03-10）
> ② 《Tensor Core 微架构深度解析》 https://zhuanlan.zhihu.com/p/2028548125518755193
> 本文件为结构化归档（含全部关键数据表、论文/工具清单），并标注与本项目 MX450 工作的对应关系。

---

## 一、文章①《GPU逆向方法总结》—— 八类方法

| 章节 | 方法 | 手段 | 可得到什么 |
|---|---|---|---|
| 0x01 | 微基准测试 | `clock`/`clock64` 计数 + 控制变量 | 流水线延迟/吞吐、调度策略、TC 特性 |
| 0x02 | ISA 二进制分析与补丁 | bit-flipping、SASS 补丁 | 指令编码格式、控制码语义、依赖屏障 |
| 0x03 | 内存层次探测 | Pointer-Chasing | 缓存大小/相联度/替换策略、TLB 层次、延迟 |
| 0x04 | 二进制插桩 | NVBit / SASSI | 运行时指令级行为 |
| 0x05 | 模拟器驱动 | GPGPU-Sim / Accel-Sim | 假设显式化、硬件-模型对比 |
| 0x06 | 侧信道与安全 | timing / contention | 调度器粒度、共享资源边界 |
| 0x07 | 功耗与热建模 | 微基准 + 回归 | 组件级功耗、功率门控粒度 |
| 0x08 | 工具与汇编器 | CuAssembler / maxas / TuringAs / DocumentSASS | SASS 级修改与优化 |

### 1.1 关键论文与数据（按代际）

| 论文 | 架构 | 关键发现 |
|---|---|---|
| Wong 2010 (ISPASS) | GT200 | 三级指令缓存 4/8/32 KB；L1 TLB 16 路全相联映射 8MB、行 512KB；L2 TLB 32MB 8 路、行 4KB |
| Jia 2018 (arXiv:1804.06826) | Volta V100 | L2 6144 KiB / 900 GB/s @877MHz；L2 物理索引；L1.5 常量缓存与 L1I 共享；**6 个依赖屏障** |
| Jia 2019 (arXiv:1903.07486) | Turing T4 | 指令编码与 Volta 相同；**双端口寄存器 bank**；**reuse flag 每条最多 4 操作数**；L2 16 路；**L1 miss/L2 hit=188 cycles，L2 miss/TLB hit=296，full miss=616** |
| Abdelkhalik 2022 (arXiv:2208.11174) | Ampere A100 | 全类型 CPI 目录；**SASS barrier 约 33 周期** |
| Arafa 2019 (arXiv:1905.08778) | 5 代 7 卡 | 低开销指令延迟特征化框架（NMSU-PEARL/GPUs-ISA-Latencies） |
| Luo 2024 (arXiv:2402.13499) | Hopper H100 | 首次特征化 **WGMMA**；**平均 L2 延迟 = L1 的 6.5 倍** |
| Luo 2025 (arXiv:2501.12084) | Hopper | TMA 1.5x；FP8 ≈ 2× FP16；DPX ≥4.75x |
| Huerta 2025 (arXiv:2503.20481, MICRO) | Ampere/Turing/Blackwell | **CGGTY 调度策略，无传统记分板、无 operand collector**；每 bank 1024-bit 读端口；RFC 每 entry 3×1024-bit；**41 控制位/warp = 寄存器堆 0.09%**；Accel-Sim MAPE 33%→13.45% |
| Jarmusch 2025a (arXiv:2512.02189) | Blackwell B200 | ResNet-50 1.85x vs H200；缓存未命中延迟 −58%；解压缩引擎 42–462 GB/s；首次特征化 TMEM |
| Jarmusch 2025b (arXiv:2507.10789) | Blackwell GB203 | L2 命中约 358 cycles（固定）；GH100 L2 65MB |
| Mei & Chu 2017 (arXiv:1509.02308) | Fermi/Kepler/Maxwell | **GPU L1 数据缓存不用 LRU** → 推翻传统 P-chase 假设 |
| Zhang 2023 (CCS) | Turing/Ampere | **MIG 不分区 L3-uTLB**；L3-uTLB 8 路 2 slice，每 slice 8 entry victim buffer |
| Sun 2023 (TPDS) | Volta/Turing/Ampere | ThC 代际：Volta 8 个/SM 4×4×4 仅 FP16；Turing +INT8/INT4/Binary；Ampere 4 个/SM 8×4×8 +BF16/TF32/稀疏 |
| Khairy 2019/2020 | Volta / 全代 | 内存模型误差 −66x；**Accel-Sim** 80 负载 1945 内核，周期误差 −79 个百分点 |
| Villa 2019 (NVBit) / Stephenson 2015 (SASSI) | 多代 | 动态 / 静态插桩框架 |
| Leng 2013 / Kandiah 2021 | GPUWattch / AccelWattch | 误差 9.9%/13.4%；AccelWattch Volta MAPE 7.5–9.2%；**功率门控粒度：芯片级 L2 / SM 级 L1 / lane 级 FPU** |
| Naghibijouybari 2017/2018 | Maxwell/Volta | 隐蔽通道 >4 Mbps；侧信道致 **CVE-2018-6260** |

### 1.2 ★ 工具清单（可直接用于 MX450）

| 工具 | 仓库 | 对我们的用途 |
|---|---|---|
| **CuAssembler** | `cloudcores/CuAssembler` | Kepler~Hopper 全代 SASS 反汇编/汇编/补丁 → **改我们自己 GEMM 的 stall/barrier/reuse 控制码** |
| **TuringAs** | `daadaada/turingas` | **Volta/Turing 的 SASS 汇编器**，直接对应 sm_75 |
| **maxas** | `NervanaSystems/maxas` | Maxwell SGEMM 达理论峰值 **98%**、比 cuBLAS 快 **4.8%** → 我们 80.9%→~95% 的模板 |
| **DocumentSASS** | `0xD0GF00D/DocumentSASS` | `LD_PRELOAD` 拦 nvdisasm，抽取 **内嵌的指令规范与调度延迟** → 省掉微基准测固定延迟 |
| **nv_isa_solver** | `kuterd/nv_isa_solver` | fuzz nvdisasm 自动生成 ISA 规范（已出 SM90a/SM89） |
| **AmpItUp** | `NoxNode/AmpItUp` | Ampere 指令编码逐位注释（方法可套 sm_75） |
| **NVBit / SASSI** | `NVlabs/NVBit` / `NVlabs/SASSI` | 插桩我们自研 kernel |
| **Accel-Sim** | `accel-sim/accel-sim-framework` | 校核微基准结论 |
| **gpu-tlb** | `0x5ec1ab/gpu-tlb` | TLB 逆向（含 pagemap/页表工具） |
| **CuAsmRL / SIP** | arXiv:2501.08071 / 2403.16863 | 自动搜 SASS 调度，实测 **+9%（最高 26%）** |
| JEB 5.31+ | 商业 | Volta~Blackwell 反编译 |

### 1.3 可直接照搬的方法论要点

1. **CuAsmRL 的教训**：基于时钟的 SASS 微基准**可能不准**（它把 IADD3 测成 2.6 cycles），
   **基于依赖的测法**（逐步降低 stall count 直到结果出错）更可靠。→ 我们测延迟时应改用此法。
2. **控制码是性能的隐藏层**：cuBLAS 的优势主要来自 SASS 级流水线管理，而非算法（Schuetze 2024 逆向 cuBLAS 结论）。
3. **TuringAs 的关键优化**：用 8 个 FFMA 交错 LDG（cuDNN 只用 2 个）→ 单项 **1.24x**。
   这与我们 v7 的"双缓冲 + 交错加载"方向一致，但它是 SASS 级精确控制。
4. **NVIDIA 专利是逆向情报源**：`US20150220341A1`（软件记分板 DEPBAR，6-bit 计数 0–63）、
   `US11080051B2`（LDGSTS 异步拷贝 + LDGDEPBAR）。

---

## 二、文章②《Tensor Core 微架构深度解析》—— ★ 直接解释我们的实测

### 2.1 基础事实

- Volta SM = 4 Sub-Core，每 Sub-Core：1 Warp Scheduler（1 instr/clk）、1 Dispatch、RF 64KB、**2 个 Tensor Core**、8 FP64、16 FP32、16 INT、4 MUFU、MIO、L0/L1 ICache
- Titan V：80 SM × 8 TC = 640 TC；1530 MHz → **理论 125 TFLOPS（FP16 混合精度）**
- PTX `wmma.mma` → 多条 **HMMA** SASS 指令

### 2.2 ★★ HMMA.884 的 SET/STEP 结构（与我们的 MX450 直接对应）

```
HMMA.884.F32.F32.STEP0 R8,  R24.reuse.COL, R22.reuse.ROW, R8;   ← 第10周期
HMMA.884.F32.F32.STEP1 R10, R24.reuse.COL, R22.reuse.ROW, R10;  ← 第12周期
HMMA.884.F32.F32.STEP2 R4,  R24.reuse.COL, R22.reuse.ROW, R4;   ← 第14周期
HMMA.884.F32.F32.STEP3 R6,  R24.COL,       R22.ROW,       R6;   ← 第18周期
```

| 事实 | 数值 |
|---|---|
| 一个 `wmma.mma` 16×16×16 混合精度 → **16 条 HMMA**（4 SET × 4 STEP） | 累积 44 周期（10→54） |
| FP16 累加模式 → **8 条 HMMA**（4 SET × 2 STEP） | 总延迟 64 周期 |
| 单个寄存器编号 = **一对相邻寄存器** | R8 代表 R8:R9 |
| `reuse` 标注 = 缓存到 **Operand Reuse Cache** | 每指令最多 4 操作数 |
| 每 SET：4×4 A × 4×8 B → 4×8 C/D；每 STEP：2×4 A × 4×4 B | 4×4=16 次基本乘加 |
| 每 TC = 16 个 FEDP 单元，4 级流水 | 寄存器带宽 6kb/2cycle/warp = 1.5kb/cycle/TC |
| **TC 不能与 INT/FP32 co-issue** | 因为它占满寄存器访问端口（3 源 × 512bit = 1536 bit/cycle） |
| 并发度 | 每 SM **仅 4 warp 可并发执行 HMMA**；每 warp 用 2 个 TC |

### 2.3 Turing 的时序表（Table I，16×16×16）

| 精度 | SET1 | SET2 | SET3 | SET4 | 总延迟 |
|---|---|---|---|---|---|
| FP32 累加 | 42 | 56 | 78 | 99 | **99 周期** |
| FP16 累加 | 44 | 52 | 60 | 74 | 74 周期 |
| INT8 | 40 | 44 | 47 | 59 | 59 周期 |

**Turing 的 16×16×16 混合精度需 99 周期，比 Volta 的 54 周期更慢**（但 Turing 支持 32×8×16 / 8×32×16 / 8×8×32 更大 tile）。

### 2.4 ★★★ 对本项目的决定性推论

**我们实测**：MX450(TU117, sm_75) 上 cuBLAS fp16 走 `cutlass_70_tensorop_f16_s884gemm`，
只有 **0.3191 TFLOPS = 时钟折算峰值的 9.9%**（FP32 FFMA 可达 99.0%）。

**文章②给出的机制解释**：
1. `s884` 就是 **`mma.m8n8k4` → `HMMA.884`** 这条路径，一个 wmma 需要 16 条 HMMA、99 周期；
2. 而 **TU117 没有物理 Tensor Core**（本项目已实测确认：FP16 5.77 vs mma 0.60，慢 9.6x），
   所以每条 HMMA.884 必须由普通 FP16/FEDP 通路**逐条模拟**；
3. HMMA 还**不能与 FP32/INT co-issue**（占用全部寄存器端口），
   → 模拟时既拿不到 TC 吞吐，又阻塞标量流水 → 只剩 ~10%。

**→ 这就是 9.9% 的完整因果链，不是"驱动没优化"，而是硬件缺失 + 指令序列不匹配。**

**由此可推**：在 MX450 上任何"走 HMMA 的库"（cuBLAS fp16 / Triton / TileLang / FlagGems）
都会撞同一堵墙（实测 Triton 0.595 / TileLang 0.538 / FlagGems 0.29-0.55x，全部印证）。
**唯一出路是我们自研的 FFMA 路径**（已 2.4872 TFLOPS）。

---

## 三、两篇文章 → 本项目下一步的映射（可执行）

| 文章方法 | 我们已做的 | 可直接升级的动作 |
|---|---|---|
| 0x01 微基准 | FMA 峰值 2.4747（99.0%）、GEMM 2.4872（80.9%）、时钟采样 | 沿用；补 **依赖式测法**（CuAsmRL 教训） |
| 0x02 SASS 分析/补丁 | 抓 kernel 名 `cutlass_70_tensorop_f16_s884gemm` | **引入 CuAssembler / TuringAs 改我们自研 GEMM 的 stall-count / barrier / reuse** |
| 0x03 内存层次 | DRAMA++ / Sudoku 测 DRAM bank 映射 | 补 **Pointer-Chasing 测 MX450 L1/L2/TLB**（2GB 显存可做） |
| 0x04 插桩 | 无 | NVBit 插桩自研 GEMM，看 stall 原因 |
| 0x05 模拟器 | 无 | Accel-Sim 校核我们的 80.9% 是否已到该配置上限 |
| 0x07 功耗/热 | 只读了 clocks_event_reasons | **AccelWattch 思路：用微基准推功率门控**（我们 power.draw=N/A，只能靠间接） |
| 0x08 工具 | 无 | **DocumentSASS 抽 nvdisasm 内嵌延迟表 → 精确知道 sm_75 每条指令延迟** |

**最高优先级**：0x08 的 DocumentSASS + 0x02 的 CuAssembler/TuringAs。
理由：maxas 在 Maxwell 用这套方法做到**理论峰值 98%、超 cuBLAS 4.8%**；
我们当前 80.9%，缺的正是 **SASS 级 stall/barrier 精确控制**（不是算法、不是 tile 尺寸）。
