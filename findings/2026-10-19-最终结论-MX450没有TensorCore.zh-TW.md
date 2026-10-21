# 🔴🔴🔴 最終結論：MX450 沒有 Tensor Core

## 一、我必須先承認一個判斷錯誤

**本專案此前的表述一直是：「Tensor Core 存在，但被限速到理論值的 5%（0.60 vs 6–12 TFLOPS）。」**

**這個判斷是錯的。**

**正確的結論是：TU117 的 SM 裡沒有 Tensor Core 硬體。
0.60 TFLOPS 是 FP16 單元執行 HMMA 指令的模擬效能，不是「被限速的 TC」。**

**這個錯誤持續了整個專案週期，直到 2026-10-19 14:30 透過連網檢索才被糾正。**

## 二、公開資料的確認（三方獨立）

### 超能網（Expreview）2019-04-24 評測，原文：

> 「GTX 1650 所用的 TU117 和之前的 TU116 一樣**去掉了 RT Core、Tensor Core**，
> SM 單元內部架構經過了修改，**原本放 Tensor Core 的位置變成了 FP16 半精度浮點運算單元**，
> 處理遊戲資料要比從前的 Pascal 架構來得更高效」

### Notebookcheck（GTX 1650 Laptop）：

> 「與更快的 RTX 2000 GPU（例如 RTX 2060）相比，**1650 沒有整合 Raytracing 或 Tensor 核心**」

### 百度百科（GeForce GTX 1650）：

> 「GTX 1650 的核心（TU117）**移除了專門用於即時光線追蹤的 RT Core 和用於 AI 超取樣（DLSS）的 Tensor Core**」

### 對照組：RTX 20 系列（TU10x）

> 「每個 SM 包含 64 個 CUDA 核心、**8 個 Tensor Cores**、一個 256KB 暫存器、4 個紋理單元」

**TU117：0 個 Tensor Core。位置被 FP16 單元取代。**

## 三、實測證據鏈（本機）

### 實驗 1 · FP32 / FP16 / mma 三路對比

```
GPU: NVIDIA GeForce MX450   SM=14   capability=7.5
矩陣 2048×2048，每次 17.2 GFLOP，迭代 20 次

FP32 (allow_tf32=False) : 2.122 TFLOPS   ← 理論 2.5，達 85%，方法正確
FP16 (PyTorch)          : 0.327 TFLOPS   ← 只有 FP32 的 15%
FP16 / FP32 = 0.15x
```

### 實驗 2 · 與歷史實測對照

| 路徑 | 實測 TFLOPS |
|---|---|
| FP32 cuBLAS（`volta_sgemm_128x64_nn`） | **2.122** |
| 手寫 PTX `mma.sync` | **0.60** |
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | **0.32** |
| **PyTorch FP16 CUDA core** | **0.327** |

**關鍵：cuBLAS 的 "TENSOR_OP" 路徑（0.32）與 FP16 CUDA core（0.327）幾乎完全相同。**

**這說明：所謂 "TENSOR_OP" 路徑，跑的就是 FP16 單元。**

### 實驗 3 · kernel 名（決定性）

**cuBLAS 日誌確認它選了 TC 演算法：**
```
algo: type=SOME TYPE; val=CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)
```

**但實際執行的 kernel 是：**
```
FP16: void cutlass::Kernel<cutlass_70_tensorop_f16_s884gemm_relu_f16_1...
                            ↑ cutlass_70          ↑ s884 = m8n8k4
FP32: volta_sgemm_128x64_nn
```

**兩個關鍵點：**

1. **`cutlass_70` = 為 SM 7.0（Volta）編譯，不是 sm75**
   → NVIDIA **沒有為 TU117 提供 sm75 的 TC kernel**
   → cuBLAS 回退到 sm70 的通用 kernel

2. **`s884` = m8n8k4 的 mma 形狀**（Volta 第一代 TC 形狀）
   → 不是 Turing 的 m16n8k8 / m16n8k16

**效能：276944 µs / 5 次 = 55 ms per matmul → 0.31 TFLOPS**

## 四、完整解釋

**為什麼 HMMA 指令能執行？**

CUDA capability 7.5 宣告支援 sm_75 的全部指令集，**包括 HMMA**。
指令能被編譯、能被解碼、能產生正確結果。

**但執行單元不是 TC。**

TU117 的 SM 裡，原本放 TC 的位置是 **FP16 單元**。
HMMA 指令被對映到 FP16 單元執行——**結果是正確的，速度是 FP16 的**。

**這就解釋了三件事：**

| 現象 | 解釋 |
|---|---|
| HMMA 能編譯執行、結果正確 | 指令集支援 |
| 效能只有 0.31–0.60 TFLOPS | 由 FP16 單元執行，不是 TC |
| cuBLAS 選 cutlass_70 而非 sm75 | NVIDIA 沒有為 TU117 出 sm75 TC kernel |
| FP16 CUDA core ≈ TENSOR_OP 效能 | 兩者走同一個執行單元 |

## 五、為什麼之前所有的「找鎖」都失敗了

**因為這個專案在找一個不存在的東西。**

回顧本專案走過的路：

| 階段 | 做了什麼 | 結果 |
|---|---|---|
| 硬體實測 | 測出 0.60 TFLOPS，判定「被限速到 5%」 | **前提就錯了** |
| GSP 韌體 | 完整反組譯 30 MB RISC-V ELF | 不含 TC 邏輯（合理，因為 TC 不存在） |
| vBIOS | 發現 FWSEC 憑證鏈驗證 | 與 TC 無關 |
| 系統 BIOS | Boot Guard 加密 | 與 TC 無關 |
| PRAMIN | 打通顯示記憶體讀寫，dump 32 MB | 顯示記憶體裡沒有 TC 配置 |
| FECS 微碼 | 完整 dump 32 KB + 反組譯 11401 行 | 只管排程與同步 |
| GPCCS 微碼 | 完整 dump 16 KB + 反組譯 6059 行 | 同上 |
| Falcon IO | 確認 host 視窗僅 4 KB | `0x20000` 段不可見 |
| nvidia.ko | 符號級逆向，46614 符號 | 五類線索全部排除，Tensor/MMA 零命中 |

**九條路徑，全部失敗——因為它們都在找「鎖」，而問題是「沒有門」。**

**現在回頭看，最有說服力的證據其實一開始就在：
`nvidia.ko` 裡 `tensor|HMMA|MMA_` 字串零命中。
如果一塊卡有 TC，驅動裡不可能一個相關符號都沒有。**

## 六、這個結論的實際意義

### 對「解鎖 TC」這個目標

**不可能。**
不是「被鎖住了打不開」，是「這個功能在矽晶片上不存在」。

**要獲得 TC，唯一的辦法是換卡。**

### 對 MX450 的真實定位

| 項目 | 真實值 |
|---|---|
| FP32 | 2.12 TFLOPS（理論 2.5，達 85%） |
| FP16（模擬） | 0.33–0.60 TFLOPS |
| **Tensor Core** | **無** |
| RT Core | 無 |
| 顯示記憶體 | 1.68 GB 可用 / 2 GB 物理 |
| 7B QLoRA 需求 | ~4 GB |

**MX450 是一塊規格完整的入門級 FP32 顯示卡，它的 FP32 效率（85%）甚至相當不錯。
它只是沒有 TC，也不可能有。**

### 對本專案的價值

**這份研究仍然有價值，但價值定位變了：**

不是「我嘗試解鎖 TC 但失敗了」，
而是「**我用九條路徑證明了 TU117 沒有 TC，並留下了完整的證據鏈與方法論**」。

**其中包括：**

- 一套可重現的 Turing 逆向工具鏈（PRAMIN、Falcon IMEM dump、nvidia.ko 符號分析）
- FECS/GPCCS 微碼的完整 dump 與反組譯（公開資料未見）
- Falcon IO 位址空間的完整模型（`HOST_IO_INDEX` 僅 6 位）
- nvidia.ko 的 GRCTX 佈局（186 個欄位全部定位）
- **一條重要的方法論：判斷「有沒有」比判斷「能不能開」更重要，且更早該做**

## 七、方法論教訓（寫給自己）

**這個專案最大的教訓：在投入大量精力尋找「解鎖方法」之前，
應該先用最便宜的方法確認「這個東西到底存不存在」。**

**最便宜的方法是什麼？**
1. 查公開規格（30 秒）—— 超能網 2019 年就寫清楚了
2. 看驅動裡有沒有相關符號（1 分鐘）—— `strings nvidia.ko | grep -i tensor` 零命中
3. 對比同類硬體的效能數量級（5 分鐘）

**這三步加起來不到 10 分鐘，能省掉整個專案的 20 小時。**

**而我們是在 20 小時後，才透過一次連網檢索發現的。**

**這就是「每步都連網搜尋」的真正價值——它不只是補充資訊，
它能防止你在一個錯誤的前提上建造一整座城堡。**

## 八、最終資料彙總

```
裝置: NVIDIA GeForce MX450 (TU117)
SM: 14    capability: 7.5    顯示記憶體: 2048 MiB (1681 MiB 可用)
驅動: 580.142    CUDA: 13.0

FP32 峰值:  2.122 TFLOPS  (volta_sgemm_128x64_nn)
FP16 峰值:  0.327 TFLOPS  (CUDA core)
mma.sync:   0.600 TFLOPS  (手寫 PTX)
cuBLAS TC:  0.320 TFLOPS  (cutlass_70_tensorop_f16_s884gemm)

Tensor Core: 不存在（TU117 移除，位置為 FP16 單元）
```

## 九、致謝與聲明

**本結論基於：**
- 3 篇獨立公開資料（超能網 / Notebookcheck / 百度百科）
- 4 組本機實測（FP32 / FP16 / mma.sync / cuBLAS TENSOR_OP）
- 1 次 cuBLAS kernel 名擷取（`cutlass_70_tensorop_f16_s884gemm`）
- 9 條逆向路徑的排除性證據

**所有實測資料可用 `tools/bench_fp16_vs_tc.py` 與 `tools/profile_kernel2.py` 重現。**

**此前 findings 中所有「TC 被限速」的表述，以本文為準。**
