# TileLang 真實試用記錄

> 使用者要求「必須真跑」。本文是 TileLang 在 MX450 上的完整實測過程，含三次失敗與最終成功。

---

## 一、為什麼試它

**搜尋結果明確寫著：**

> **"tile-lang supports CUDA devices with architectures including Ampere (sm_80+),
> Turing (sm_75), and Volta (sm_70)."**

**這是唯一一個**明確聲明支援 sm_75** 的高效能 GEMM DSL。
如果能跑，就是對比本機手寫 kernel 的最有力參照。**

---

## 二、四次嘗試，三次失敗

### 失敗 1 · Target 偵測失敗

```
pip install tilelang          # 成功，0.1.15
```

```python
kernel = tilelang.compile(func, out_idx=[2])
```

```
FAIL: No registered target detector found an available target.
```

**原因：TVM 沒有自動識別 CUDA 後端。**

**修復：顯式指定 `target="cuda"`。**

### 失敗 2 · 缺 CUDA_HOME

```
FAIL: Failed to automatically detect CUDA installation.
      Please set the CUDA_HOME environment variable manually.
```

**檢查發現：pip 的 `nvidia-cuda-nvcc-cu12` 包只有 `ptxas`，沒有 `nvcc`。**

```
nvidia/cuda_nvcc/bin/ptxas      ← 只有這個
nvidia/cuda_nvcc/include/...    ← 只有標頭檔
```

**這是 pip 包的固有缺陷：它不提供完整 nvcc 編譯器。**

**修復：從 NVIDIA 官方 apt 源裝獨立元件（不是整個 toolkit）。**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-nvcc-12-9      # 只裝編譯器，約 27 MB
```

**結果：`/usr/local/cuda-12.9/bin/nvcc`（27 MB），版本 12.9.86。**

**注意：`cuda-nvcc-12-1` 不存在，可用的是 12-5 / 12-6 / 12-8 / 12-9 / 13-0～13-3。**

### 失敗 3 · DLPack allocator

```
FAIL: TVM-FFI callee-allocated outputs require Torch's DLPack allocator exchange API.
      Install a compatible torch-c-dlpack-ext or use a supported PyTorch build.
```

**原因：`out_idx=[2]` 讓 TileLang 自己分配輸出張量，需要 `torch-c-dlpack-ext`。**

**修復（比重裝包更快）：改用 caller-allocated 輸出。**

```python
# 原本
kernel = tilelang.compile(func, out_idx=[2], target="cuda")
c = kernel(a, b)

# 改成
kernel = tilelang.compile(func, target="cuda")
c = torch.empty(M, N, device="cuda", dtype=torch.float16)
kernel(a, b, c)
```

### 成功 · 正確性 + 效能

```
正確性 (512³):
  max abs err = 0.06250  (相對 5.56e-04)   PASS

效能 (2048³, float16, 強制同步):
  (128,128,32)  編譯 1.5s   0.527 TFLOPS  (32.6 ms)
  (64,128,32)   編譯 1.4s   0.538 TFLOPS  (31.9 ms)   ← 最佳
  (128,128,64)  FAIL  共享記憶體 98304 > 上限
  (128,256,32)  FAIL  共享記憶體 73728 > 上限
```

---

## 三、一個硬體約束被暴露出來

**兩個大 tile 配置失敗，原因是共享記憶體超限：**

```
Failed to set the allowed dynamic shared memory size to 98304   (96 KB)
Failed to set the allowed dynamic shared memory size to 73728   (72 KB)
```

**MX450（Turing）的 SM 共享記憶體上限是 64 KB（需 opt-in 才到 96 KB）。**

**TileLang 沒有為 sm_75 做這個 opt-in，所以 block_K=64 這類配置直接失敗。**

**這意味著：在這塊卡上，TileLang 能用的 tile 配置被顯著壓縮。**

---

## 四、完整橫向對比（全部強制同步）

| 實作 | TFLOPS | 相對手寫 | 走的路徑 |
|---|---|---|---|
| cuBLAS `TENSOR_OP` | 0.320 | 0.20x | HMMA（模擬） |
| tinygrad（同步後） | 0.347 | 0.22x | HMMA（推測） |
| **TileLang（最佳）** | **0.538** | **0.34x** | **HMMA（cute/CUTLASS）** |
| **本機手寫 CUDA C（FFMA）** | **1.603** | **1.00x** | **FFMA** |
| cuBLAS FP32 | 2.122 | 1.32x | FFMA |

**關鍵觀察：**

**所有走 HMMA 的實作（cuBLAS TENSOR_OP / tinygrad / TileLang）都在 0.32–0.54 區間。**

**所有走 FFMA 的實作（手寫 / cuBLAS FP32）都在 1.60–2.12 區間。**

**這不是巧合，是 TU117 沒有真 TC 的直接後果。**

---

## 五、結論

### TileLang 能用，但不適合這塊卡

| 方面 | 評價 |
|---|---|
| sm_75 支援 | ✅ 官方聲明支援，實測能編譯執行 |
| 正確性 | ✅ PASS（相對誤差 5.56e-04，FP16 合理） |
| 效能 | ❌ 0.538，是手寫的 34% |
| 大 tile | ❌ 共享記憶體 64 KB 限制，多個配置失敗 |
| 安裝門檻 | ⚠️ 需要完整 nvcc（pip 包不夠） |

**根本原因：`T.gemm` 在 NVIDIA 上 dispatch 到 cute/CUTLASS，
而 CUTLASS 在 sm_75 上預設走 HMMA。TU117 的 HMMA 是模擬的，所以慢。**

**如果 TileLang 提供「停用 tensor core」的選項（類似 `tl.dot` 的 input_precision），
理論上能走 FFMA。但目前版本沒有暴露這個開關。**

### 這三次失敗 + 一次成功，共同證明了一件事

**在 TU117 上，「用高層 DSL」不是最優解。
唯一能拿到 1.6 TFLOPS 的方式，是自己寫 CUDA C 並強制走 FFMA。**

**本專案的 `sw_tensor_core_v2.py` 就是這個結論的產物。**

---

## 六、環境記錄

```
tilelang    0.1.15
torch       2.3.1+cu121
nvcc        12.9.86（從 NVIDIA apt 源單獨安裝）
CUDA_HOME   /usr/local/cuda-12.9
GPU         MX450, 14 SM, CC 7.5
共享記憶體上限 64 KB（96 KB 需 opt-in）
```

**重現：**
```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
python test_tilelang.py
```

**檔案：`full-unlock/03-toolkit/test_tilelang.py`**
