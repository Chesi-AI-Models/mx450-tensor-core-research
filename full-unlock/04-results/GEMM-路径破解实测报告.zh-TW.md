# MX450 (TU117) GEMM 路徑破解 —— 實測報告

> 全部資料來自 LYLNAS 實機（NVIDIA GeForce MX450 / TU117 / CC 7.5 / 14 SM / 1.64 GiB）
> 環境：torch 2.3.1+cu121，CUDA 12.1，驅動 nvidia-open 580.142
> 時間：2026-10-19

---

## 0. 一句話結論

**MX450 上永遠不要用 fp16 直接做 GEMM。**
cuBLAS 為 CC 7.5 選擇的是為 sm_70 優化的 `mma.sync.m8n8k4`（SASS: `HMMA.884`），
TU117 沒有能執行它的硬體，只能模擬 → 實測慢 **7.5 倍**。
改走 fp32 中轉或分塊，立刻拿回 **3.3x ~ 4.5x**，其中 fp32 中轉**精度零損失**。

---

## 1. 根因取證：kernel 名直接定案

用 `torch.profiler` 抓 N=1024 的實際 kernel：

| dtype | 實測耗時 | kernel 名 |
|---|---|---|
| **fp16** | **20318 us** | `cutlass::Kernel<cutlass_70_tensorop_f16_s884gemm_relu_f16_128x64_nn_align8>` |
| **fp32** | **2685 us** | `volta_sgemm_128x64_nn` |
| **bf16** | 4239 us | `magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, ...>` |

**`s884` 的含義**：`mma.sync.aligned.m8n8k4`，Volta 時代的 warp 級 MMA 指令。
SASS 對映表明確寫：`mma.m8n8k4.f16 (Volta form) → HMMA.884.F32.F32.STEP{0..3}`，作用域 **sm_75**。

**NVIDIA 官方 PTX 文件原話**：

> Note: `mma.sync.m8n8k4` is optimized for target architecture `sm_70` and **may have
> substantially reduced performance on other target architectures**.

cuBLAS 在 CC 7.5 上偏偏選了這條被官方警告的指令。這就是 7.5 倍損失的全部來源。

---

## 2. 三方獨立印證（不是我們一台機器的問題）

| 來源 | 硬體 | 結論 |
|---|---|---|
| 本報告（實機） | MX450 / TU117 | fp16 比 fp32 慢 **7.45x**（N=1024） |
| GitHub `msraaghavan/HumanDiT` hardware_notes.md | **GTX 1650 Ti / TU117** | `GEMM 4096² fp32 1.72 TFLOPS, fp16 0.41 TFLOPS (4× slower: no tensor cores, no fast HGEMM path)`；結論 **"precision policy uses fp32 on sm_75"**；`bf16 emulated; not useful`；FlashAttention 需 sm_80+ |
| GitHub `jamesburton/dotLLM` GPU.md | 通用 | 「On pre-Volta GPUs cuBLAS FP16 GEMM falls back to CUDA cores」 |
| CSDN 問答 | MX450 | 「FP16 實際走 FP32 單元截斷模擬路徑……持續吞吐驟降至約 **0.25 TFLOPS**」 |

我們實測 fp16 各尺寸 **0.239 ~ 0.351 TFLOPS**，與 CSDN 的「0.25 TFLOPS」吻合。

---

## 3. 尺寸掃描：fp32 的甜區與斷崖

| N | fp16 TFLOPS | fp16 ms | fp32 TFLOPS | fp32 ms | fp32/fp16 |
|---|---|---|---|---|---|
| 256 | 0.2389 | 0.140 | 0.5993 | 0.056 | 2.51x |
| 512 | 0.2472 | 1.086 | 1.3148 | 0.204 | 5.32x |
| **1024** | 0.3183 | 6.746 | **2.3714** | 0.906 | **7.45x** |
| **2048** | 0.3268 | 52.575 | **2.1098** | 8.143 | **6.46x** |
| 3072 | 0.3514 | 165.022 | 0.9330 | 62.145 | 2.66x |
| 4096 | 0.3490 | 393.815 | **0.3533** | 388.974 | 1.01x |

**兩個關鍵事實：**
1. fp16 的 TFLOPS 幾乎不隨尺寸增長（0.24 → 0.35），說明瓶頸固定在指令模擬，不是排程／佔用率問題
2. **fp32 在 1024/2048 是甜區（2.1-2.4 TFLOPS，達物理峰值 3.06 的 77%），3072 掉一半，4096 掉到 0.35（斷崖）**

bf16 對照：1024 → 1.356 TFLOPS，2048 → 1.299。**比 fp16 快 4.3x**，
因為 cuBLAS 用 MAGMA `sgemmEx` 把 bf16 升到 fp32 計算 —— 反而繞開了壞路徑。

---

## 4. 方案一：fp16 → fp32 → GEMM → fp16（精度無損）

```python
def mx450_mm_fp16(a, b):          # a, b: fp16 CUDA tensors
    return torch.mm(a.float(), b.float()).half()
```

| N | fp16 直連 | **fp16→fp32→mm→fp16** | 純 fp32 | 加速比 | max_abs_err |
|---|---|---|---|---|---|
| 1024 | 6.760 ms | **1.828 ms** | 1.621 ms | **3.70x** | **0.000000** |
| 2048 | 54.014 ms | **16.199 ms** | 15.778 ms | **3.33x** | **0.000000** |
| 3072 | 167.870 ms | 98.366 ms | 97.255 ms | 1.71x | 0.000000 |
| 4096 | 398.797 ms | 628.999 ms | 628.690 ms | **0.63x** ⚠ | 0.000000 |

**注意**：4096 反而變慢，因為此時 fp32 自己也掉進斷崖。**4096 必須配合方案二分塊。**

### 備選：fp16 → bf16 中轉（更快但掉精度）

| N | fp16 直連 | fp16→bf16 | 加速 | 相對誤差 |
|---|---|---|---|---|
| 1024 | 6.780 ms | **1.611 ms** | 4.21x | **0.38%** |
| 2048 | 54.273 ms | **14.297 ms** | 3.80x | **0.37%** |

bf16 只有 8 位尾數，誤差 0.38%。**除非能容忍這個精度，否則用 fp32 中轉。**

---

## 5. 方案二：分塊 GEMM 繞開 4096 斷崖（本報告最大收益）

原理：cuBLAS 在 N≥3072 時 heuristic 選到爛 kernel；把大 GEMM 拆成落在甜區的小塊。

```python
def mx450_blocked_mm(a, b, bs=512):
    n = a.shape[0]
    c = torch.empty(n, n, dtype=a.dtype, device=a.device)
    for i in range(0, n, bs):
        ai = a[i:i+bs, :]
        for j in range(0, n, bs):
            bj = b[:, j:j+bs].contiguous()
            c[i:i+bs, j:j+bs] = torch.mm(ai, bj)
    return c
```

### fp32 分塊實測

| N | block | 耗時 | TFLOPS | vs 整塊 |
|---|---|---|---|---|
| 2048 | whole | 8.305 ms | 2.0687 | 1.00x |
| 2048 | 512 | 13.357 ms | 1.2862 | 0.62x |
| 2048 | 1024 | 9.110 ms | 1.8857 | 0.91x |
| 2048 | 2048 | 8.682 ms | 1.9787 | 0.96x |
| 3072 | whole | 62.274 ms | 0.9311 | 1.00x |
| 3072 | 512 | 45.000 ms | 1.2885 | **1.38x** |
| 3072 | **1024** | **42.149 ms** | **1.3757** | **1.48x** |
| 4096 | whole | 369.192 ms | 0.3723 | 1.00x |
| 4096 | **512** | **82.459 ms** | **1.6668** | **4.48x** |
| 4096 | 1024 | 95.256 ms | 1.4428 | 3.88x |
| 4096 | 2048 | 134.701 ms | 1.0203 | 2.74x |

**N=4096，block=512 → 369 ms 降到 82 ms，4.48 倍。**

分塊的 `err` 在 1e-3 量級，是 fp32 累加順序變化導致的正常浮點誤差
（數值量級 ~1000，相對誤差 ~1e-6），不是錯誤。

### fp16 分塊：無效，符合預期

| N | whole | block=1024 | 效果 |
|---|---|---|---|
| 2048 | 54.272 ms | 54.925 ms | 0.99x |
| 4096 | 400.736 ms | 438.842 ms | 0.91x |

**證明 fp16 的瓶頸是 HMMA 模擬，不是 cuBLAS heuristic** —— 分塊救不了，只能換 dtype。

---

## 6. 可直接落地的四條規則

| # | 規則 | 依據 | 收益 |
|---|---|---|---|
| **1** | **GEMM 裡絕不用 fp16 直連** | cutlass_70_tensorop_f16_s884 + PTX 文件警告 | 避免 7.5x 損失 |
| **2** | **fp16 需求 → 轉 fp32 算 → 轉回** | 1024/2048 實測 3.3-3.7x，err=0 | 3.3-3.7x |
| **3** | **N≥3072 時用分塊 GEMM（bs=512）** | 4096 實測 4.48x | 1.4-4.5x |
| **4** | **bf16 在 Turing 上是 fp32 模擬，但精度只有 8 位尾數** | magma_sgemmEx + rel=0.38% | 4.2x 但掉精度 |

### 組合策略（推薦）

```python
def mx450_gemm(a, b):
    """MX450 上的 GEMM 最優路徑（fp16 輸入 → fp32 計算 → fp16 輸出）"""
    n = a.shape[-1]
    fp32 = a.dtype in (torch.float16, torch.bfloat16)
    x, y = (a.float(), b.float()) if fp32 else (a, b)
    bs = 512 if x.shape[0] >= 3072 else 0
    c = mx450_blocked_mm(x, y, bs) if bs else torch.mm(x, y)
    return c.half() if fp32 else c
```

### 附：框架層設定

```python
# Turing 上 autocast 的 bf16/fp16 GEMM 會掉進壞路徑，訓練改用 fp32
torch.backends.cuda.matmul.allow_tf32 = False   # 無 TC，TF32 無意義
# 實測：allow_fp16_reduced_precision_reduction 與 allow_tf32 對 fp16 mm 均無效（1.00x）
```

---

## 7. 與其他成果的關係

| 成果 | TFLOPS | 相對 cuBLAS fp16 |
|---|---|---|
| cuBLAS fp16（壞路徑） | 0.32 | 1.0x |
| cuBLAS fp32（甜區） | 2.37 | 7.5x |
| **自研軟體 Tensor Core** | **1.603** | **5.0x** |
| 理論物理峰值（FP32 FMA） | 3.06 | 9.6x |

**結論**：自研軟體 TC（1.603）仍顯著優於 cuBLAS fp16（0.32），
但**不如 cuBLAS fp32 甜區（2.37）**。
在 N=1024/2048 的場景下，**直接用 cuBLAS fp32 就是當前最優解**，
自研 kernel 的價值在於 fp16 輸入輸出且必須保持 fp16 精度的場合。

---

## 8. 重現腳本

- `full-unlock/04-results/cublas_path_study.py` —— 尺寸掃描 + 混合路徑 + kernel 名 + 開關
- `full-unlock/04-results/blocked_gemm_study.py` —— 分塊 GEMM + bf16 中轉

執行：
```bash
python nas_exec.py NVDIA-MX450-pojie/full-unlock/04-results/blocked_gemm_study.py
```
