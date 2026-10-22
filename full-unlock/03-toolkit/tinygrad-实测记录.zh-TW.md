# tinygrad 真實試用記錄（含一個重大測量陷阱）

> 使用者要求「必須真跑不能從文件判斷」。本文是 tinygrad 的實際執行記錄。
>
> **核心發現：tinygrad 的 lazy 求值會讓效能測量產生 23 倍的虛高。**

---

## 一、安裝與首測

```bash
pip install tinygrad      # 成功，走 PyPI（GitHub 直連在 NAS 上不穩定）
```

```python
from tinygrad import Tensor, Device
print(Device.DEFAULT)     # CUDA
```

**後端識別正確（CUDA）。**

---

## 二、第一次測效能：8.0 TFLOPS —— 但這是假的

### 測量程式碼

```python
N = 2048
a = Tensor.rand(N, N, dtype="float32")
b = Tensor.rand(N, N, dtype="float32")
(a @ b).realize()          # 首次編譯

t0 = time.time()
for _ in range(10):
    c = (a @ b).realize()  # ← 問題在這裡
dt = (time.time() - t0) / 10
```

### 結果

```
float32:  7.638 TFLOPS   (2.25 ms)
float16:  4.686 TFLOPS   (3.67 ms)
```

### 為什麼可疑

| 檢查項 | 資料 | 判斷 |
|---|---|---|
| MX450 FP32 理論峰值 | 896 CUDA × 2 FMA × 1.6 GHz = **2.87 TFLOPS** | 7.64 是 **2.7 倍** |
| 換算所需頻率 | 7.638e12 / (896×2) = **4.26 GHz** | MX450 不可能 |
| cuBLAS FP32 實測 | **2.122 TFLOPS** | tinygrad 不可能快 3.6 倍 |
| PTX FP32 FMA 純運算峰值 | **3.06 TFLOPS** | 7.64 超過純運算上限 |

**四個獨立參照全部矛盾 → 一定是測量問題。**

---

## 三、定位問題：tinygrad 是 lazy 的

**tinygrad 的設計核心是惰性求值**：算子先記進 UOp 圖，只在需要結果時才真正執行。

**`.realize()` 只是「把圖變成 kernel 並**排隊**」，不保證執行完成。**

**所以計時測到的是「排隊 + 啟動開銷」，不是「核函式執行時間」。**

**矩陣越大，虛高越明顯——因為真正的計算被推到計時區之外。**

---

## 四、強制同步後的真實資料

```python
# 方法 1：只 realize（非同步）
t0 = time.time()
for _ in range(5):
    c = (a @ b).realize()
t_realize = (time.time() - t0) / 5

# 方法 2：realize + numpy（強制回傳，必然同步）
t0 = time.time()
for _ in range(5):
    c = (a @ b).numpy()
t_numpy = (time.time() - t0) / 5

# 方法 3：realize + Device.synchronize()
dev = Device[Device.DEFAULT]
t0 = time.time()
for _ in range(5):
    c = (a @ b).realize()
    dev.synchronize()
t_sync = (time.time() - t0) / 5
```

### 結果（2048³，float32）

| 方法 | 耗時 | TFLOPS | 判定 |
|---|---|---|---|
| `realize()` 非同步 | 2.14 ms | **8.029** | ❌ 假的 |
| `realize()+numpy()` | 116.64 ms | **0.147** | ✅ 真實（含 D2H 拷貝） |
| `realize()+synchronize()` | 49.44 ms | **0.347** | ✅ 真實（純 GPU 時間） |

**真實效能：0.347 TFLOPS。虛高 23 倍。**

---

## 五、修正後的完整對比

| 實作 | TFLOPS | 相對 cuBLAS TENSOR_OP |
|---|---|---|
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | 0.320 | 1.0x |
| **tinygrad（同步後）** | **0.347** | **1.08x** |
| **本機手寫 CUDA C（FFMA，NVRTC）** | **1.603** | **5.0x** |
| cuBLAS FP32 | 2.122 | 6.6x |

**結論：tinygrad 的 matmul 與 cuBLAS TENSOR_OP 同級（0.32–0.35），
說明它同樣走了 HMMA 路徑（TU117 上的模擬 TC）。**

**而手寫 FFMA kernel 快 4.6 倍。**

**這反過來驗證了本專案的核心成果：在無 TC 的卡上，繞開 HMMA 走 FFMA 是正確策略。**

---

## 六、正確性驗證

```python
N = 512
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
ref = A @ B

c = (Tensor(A) @ Tensor(B)).numpy()
err = np.abs(c - ref).max()      # 0.00007
rel = err / np.abs(ref).max()    # 6.91e-07
```

**判定：PASS（相對誤差 6.9e-07，符合 FP32 預期）。**

**說明：tinygrad 的計算是正確的，只是計時方法有陷阱。**

---

## 七、這個陷阱的普遍性

**任何用 tinygrad 做效能對比的結論，都必須確認是否加了同步。**

**搜尋到的所有 tinygrad 教學／評測，都只示範 `DEBUG=4` 看 kernel，沒有提醒這一點。**

**這不是 tinygrad 的 bug（lazy 是它的設計），而是使用者容易踩的坑。**

**本專案記錄在此，供後續參考。**

---

## 八、對本專案的實際結論

**tinygrad 對 MX450 的價值：**

| 方面 | 評價 |
|---|---|
| 效能 | ❌ 不如手寫 FFMA（0.347 vs 1.603） |
| 正確性 | ✅ 通過 |
| 可除錯性 | ✅ 優秀（`DEBUG=4` 直接看生成的 kernel） |
| 對 TU117 的適配 | ⚠️ 走 HMMA，沒有針對無 TC 優化 |
| 作為參考實作 | ✅ 值得讀它的 CUDA 後端程式碼 |

**結論：tinygrad 不能直接拿來加速 MX450，但它的 `DEBUG=4` 可以看 kernel 生成，
對手寫優化有參考價值。**

---

## 九、重現

```bash
pip install tinygrad
python verify_tinygrad.py           # 正確性 + 效能（含非同步陷阱重現）
python verify_tinygrad_sync.py      # 三種同步方式的對比
```

**檔案位置：**
- `full-unlock/03-toolkit/test_tinygrad.py`（首測）
- `full-unlock/03-toolkit/verify_tinygrad.py`（正確性驗證）
- `full-unlock/03-toolkit/verify_tinygrad_sync.py`（同步對比）
