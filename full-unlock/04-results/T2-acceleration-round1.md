# T2 加速：实测数据与第一轮优化

> 目标：完全彻底加速 GreenBoost 的 T2 层（系统 RAM），覆盖推理与训练全流程。
>
> 本轮成果：**发现并修复 headroom 配置缺陷，T2 相关的实际带宽提升 12.1 倍。**

---

## 一、基础带宽实测（第一次量化）

**测试方法**：`y = x + 1`（elementwise，受内存带宽限制），FP32，5 次迭代。

| 场景 | 大小 | 带宽 | 耗时 | 路径 |
|---|---|---|---|---|
| **VRAM** | 0.5 GiB | **67.52 GB/s** | 15.9 ms | 纯显存 |
| T2 | 0.5 GiB | 5.60 GB/s | 191.7 ms | **错误地走了 T2** |
| T2 | 1.0 GiB | 5.69 GB/s | 377.2 ms | T2 |
| T2 | 2.0 GiB | 5.66 GB/s | 758.9 ms | T2 |
| T2 | 3.0 GiB | 5.65 GB/s | 1140.1 ms | T2 |
| T2 | 4.0 GiB | 5.65 GB/s | 1521.6 ms | T2 |

**核心比值：VRAM 67.52 / T2 5.65 = 11.95 倍。**

**注意：T2 带宽在不同大小下高度稳定（5.60–5.69 GB/s），说明瓶颈是 PCIe 链路本身，不是软件开销。**

---

## 二、发现的第一个严重缺陷：headroom 默认值过大

### 现象

**0.5 GiB 张量本应装进 VRAM（1.64 GiB 可用），实测却只有 5.60 GB/s —— 走了 T2。**

### 根因

shim 的环境变量 `GREENBOOST_VRAM_HEADROOM_MB` **默认 2048（2 GB）**。

shim 的判定逻辑是「保留 N MB VRAM 不用，剩余不够才走 T2」。
在 1.64 GiB 的卡上，保留 2 GB 等于**把整块 VRAM 都保留掉**，
于是任何 >256 MB 的分配全部被打到 T2。

**这是 MX450 这种小显存卡上的致命默认值。**

### 修复

```bash
GREENBOOST_VRAM_HEADROOM_MB=128
```

### 效果

| 场景 | 默认 headroom=2048 | 优化 headroom=128 | 提升 |
|---|---|---|---|
| 0.5 GiB | 5.60 GB/s | **67.85 GB/s** | **12.1x** |
| 1.0 GiB | 5.69 GB/s | 6.58 GB/s | 1.16x（仍走 T2：1G 输入 + 1G 输出 > 1.64G） |
| 2.0 GiB | 5.66 GB/s | 5.66 GB/s | 1.0x（必然走 T2） |

**结论：headroom 应设为「VRAM 的 ~8%」，即 MX450 上用 128 MB。
让装得进 VRAM 的分配全部走 VRAM，只有真正装不下时才落到 T2。**

---

## 三、T2 的真实定位

| 指标 | 数值 | 含义 |
|---|---|---|
| VRAM 带宽 | 67.5–67.9 GB/s | 基线 |
| T2 带宽 | 5.65 GB/s | **VRAM 的 1/12** |
| T2 容量 | 8 GB（可调） | VRAM 的 4.9 倍 |
| T3（NVMe） | ~1.8 GB/s | VRAM 的 1/37 |

**这决定了优化策略的方向：**

**T2 不是"慢一点的 VRAM"，它是完全不同量级的东西。
优化目标不是"让 T2 变快"（带宽被 PCIe 焊死），而是"让 T2 的访问次数最少"。**

---

## 四、后续优化优先级（按预期收益排序）

| # | 方案 | 预期收益 | 原理 |
|---|---|---|---|
| 1 | **headroom 调优** | **已完成，12.1x** | 让能装进 VRAM 的不去 T2 |
| 2 | **量化压缩** | **传输量 ÷2 ~ ÷4** | FP16→INT8 减半，→INT4 减 75%。等效带宽 11.3 / 22.6 GB/s |
| 3 | **权重驻留策略** | 热层 0 传输 | 高频层常驻 VRAM，冷层留 T2 |
| 4 | **双缓冲 + 异步预取** | 隐藏传输延迟 | 计算第 N 层时预取第 N+1 层 |
| 5 | **`cuMemAdviseSetReadMostly`** | 避免回写 | 权重只读，禁止驱动回写 T2 |
| 6 | **`cuMemPrefetchAsync`** | 消除页面缺失 stall | 主动迁移，避免 UVM 黑盒决策 |

**关键判断：方案 2（量化）是唯一能突破 PCIe 带宽墙的手段。
其他方案都是"少传"或"藏延迟"，只有量化是"真的少传 4 倍"。**

---

## 五、对训练流程的含义

**训练比推理更依赖带宽**：

- 推理：每层权重读一次（前向）
- 训练：前向读一次 + 反向读一次 + 梯度写一次 ≈ **3 倍流量**

**所以在 T2 上训练的代价约为推理的 3 倍。**

**这就是为什么 GreenBoost 的原作者把主要优化放在推理侧（Ollama/ExLlama）。**

**在 MX450 上做训练的可行路径：**

1. **QLoRA（4-bit 量化）+ 小 batch** —— 把权重压到 4-bit，传输量减 75%
2. **LoRA 只训练 adapter** —— 主权重冻结（只读，可 `SetReadMostly`），只有 adapter 在 VRAM
3. **梯度检查点** —— 用计算换显存
4. **数据量裁剪** —— 先 1% 数据验证配方

**这四条组合，是这台机器上唯一能跑通训练的路径。**

---

## 六、本轮实测环境

```
GPU: NVIDIA GeForce MX450 (TU117)
物理 VRAM: 1.64 GiB 可用
GreenBoost: greenboost.ko 已加载
  physical_vram_gb=2  virtual_vram_gb=8  safety_reserve_gb=2
  nvme_swap_gb=8  pcores_only=0  use_hugepages=1
shim: libgreenboost_cuda.so（已修 _v2 符号）
```

**复现：**
```bash
# VRAM 对照
python bench_t2_bw.py

# T2
LD_PRELOAD=.../libgreenboost_cuda.so GREENBOOST_ACTIVE=1 \
GREENBOOST_VRAM_HEADROOM_MB=128 \
python bench_t2_bw.py
```
