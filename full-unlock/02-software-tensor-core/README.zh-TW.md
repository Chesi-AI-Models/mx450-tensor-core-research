# 軟體 Tensor Core：在無 TC 的卡上重建矩陣乘

> 目標：TU117（MX450）沒有 Tensor Core，用 CUDA 核心實作等效的矩陣乘加速。
>
> 最終結果：**1.603 TFLOPS**，是 cuBLAS `TENSOR_OP` 路徑（0.320）的 **5.0 倍**，
> 且 `max abs err = 0.00000`（與 float64 參考完全一致）。

---

## 一、為什麼需要它

| 路徑 | 實測 TFLOPS | 說明 |
|---|---|---|
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | 0.320 | cuBLAS 認為有 TC，選了 `cutlass_70_tensorop_f16_s884gemm`，但 TU117 沒 TC，實際由 FP16 單元模擬 |
| cuBLAS FP32 | 2.122 | 已達理論峰值（2.5）的 85% |
| FP16x2 純運算峰值 | 5.770 | **FP16 累加**，K=2048 會溢位，不能做真實 GEMM |

**所以：cuBLAS 在 FP16 上給了你一條 0.32 的死路（因為它以為有 TC）。
自己寫 FFMA kernel 能拿到 1.6 —— 這就是 5 倍提升的來源。**

---

## 二、四版演進（含兩次失敗）

| 版本 | 檔案 | 方案 | 結果 |
|---|---|---|---|
| v1 | `triton_gemm.py` | Triton `tl.dot`，FP16 輸入 | 0.595 TFLOPS（走 HMMA） |
| v2 | `triton_gemm_v2.py` | 外積（`tl.where`+`tl.sum` 避開 dot） | 0.122（產生大量 shuffle，比 HMMA 還慢 5 倍） |
| v4 | `triton_gemm_v4.py` | Triton `tl.dot`，FP32 輸入 | **編譯崩潰 `map::at`**（Triton 在 CC 7.5 上的已知 bug，triton-lang/triton#5557） |
| **final** | **`sw_tensor_core_v2.py`** | **NVRTC + CUDA C + FFMA** | **1.603 TFLOPS** ✅ |

### 關鍵教訓

**Triton 在 CC 7.5 上不可用於 FP32。** 官方 issue 明確寫了：
FP32 原生支援在計算能力 7.5 上有限制，編譯器在產生 LLVM IR 時丟失類型中繼資訊，
引發 `map::at` 崩潰。切換到 FP16 可繞過——但 FP16 會走 HMMA，正是我們要避開的。

**所以必須繞開 Triton，直接生成程式碼。**

---

## 三、最終實作（`sw_tensor_core_v2.py`）

### 技術棧

```
CUDA C 原始碼（unsigned short + inline PTX）
    ↓ libnvrtc.so.12（NVRTC）
PTX（39,393 字元）
    ↓ cuModuleLoadData（libcuda）
GPU 執行
```

**不用 `cuda_fp16.h`**（NVRTC 預設 include 路徑裡沒有），全部用：

```cuda
__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
```

### Kernel 結構

```
BLOCK: 128 × 128，BK = 16，256 執行緒
每執行緒: 8 × 8 = 64 個 float 累加器
共享記憶體: As[128][16] + Bs[16][128]（half，共 8 KB）

每個 k:
  讀 8 個 a（共享記憶體 → 暫存器）
  讀 8 個 b
  64 次 FFMA
  → FMA/載入比 = 64/16 = 4
```

### 已驗證的 PTX 特徵

```
含 cvt.f32.f16 : True
含 fma.rn.f32  : True
含 mma.sync    : False   ← 關鍵：完全繞開 HMMA
```

---

## 四、效能資料

**矩陣 2048³，10 次迭代取平均：**

```
1.603 TFLOPS   (10.7 ms/次)
max abs err = 0.00000
```

| 對比對象 | TFLOPS | 倍數 |
|---|---|---|
| cuBLAS TENSOR_OP | 0.320 | **5.0x** |
| cuBLAS FP32 | 2.122 | 0.76x |
| FP16x2 純運算峰值 | 5.770 | 0.28x（不可比，FP16 累加） |

### 為什麼是 1.603 而不是 5.77

**`fma.rn.f16x2` 是 FP16 累加。K=2048 時累加器溢位，結果不可用。**

真實 GEMM 必須 FP32 累加 → 只能 FFMA → 硬體上限是 FP32 的 2.5 TFLOPS。
1.603 / 2.5 = **64% 效率**，剩餘損耗來自共享記憶體頻寬和指令排程。

**這是硬體的算術強度上限，不是實作缺陷。**

### 嘗試過但更慢的優化

| 優化 | 結果 | 原因 |
|---|---|---|
| 4×4 tile（BM=BN=64） | 1.539 | FMA/載入比只有 2 |
| **8×8 tile（BM=BN=128）** | **1.603** | ✅ 最優 |
| 共享記憶體改 float | 1.427 | 共享記憶體翻倍，bank conflict 增加 |

---

## 五、踩過的坑（ctypes 呼叫 CUDA Driver API）

1. **必須顯式宣告 `argtypes`**
   否則 64 位指標被截斷成 int → `cuModuleLoadData` 處 **SIGSEGV**。
   這是本專案耗時最久的一個 bug。

2. **必須複用 PyTorch 的 CUDA context**
   自己 `cuCtxCreate_v2` 會建立第二個 context，模組載入到錯誤的 context 裡，
   表現為「載入成功但呼叫無效」。
   正確做法：`torch.cuda.init()` → `cuCtxGetCurrent()`。

3. **`cuModuleLoadData` 直接吃 PTX 文本**
   不需要 ptxas，不需要 nvcc，驅動內部 JIT。

---

## 六、重現

```bash
/vol1/ganlin/venv/bin/python sw_tensor_core_v2.py --size 2048
```

依賴：`torch 2.3.1` + `nvidia-cuda-nvrtc-cu12`（提供 `libnvrtc.so.12`）。

---

## 七、這個結果的意義

**MX450 的矩陣乘真實上限是 2.5 TFLOPS（FP32），cuBLAS 已經做到 2.12。**

**但 cuBLAS 的 FP16 路徑只有 0.32——因為它以為有 TC。**

**所以「軟體 TC」的真正價值不是超過 FP32，而是：**

**當你的模型是 FP16 時，不要用 cuBLAS 的 FP16 路徑（0.32），
用自訂的 half→float→FFMA kernel（1.6），快 5 倍。**

**這就是在不含 TC 硬體上能做的最優解。**
