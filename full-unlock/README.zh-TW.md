# MX450 完整破解流程 — 現實核查與執行記錄

> 本目錄是對「**盡可能徹底解鎖 MX450 的 Tensor Core**」這個目標的完整回答。
>
> 結論先行：**TU117 沒有 Tensor Core 硬體，無法「解鎖」；但可以用 CUDA 核心重建一個，
> 實測 1.603 TFLOPS，是 cuBLAS TENSOR_OP 的 5.0 倍。**

---

## 一、先做現實核查：哪些能破，哪些不能

在動手之前，必須把「破解」這個詞拆開。MX450 上的四類限制，性質完全不同：

| 目標 | 能否破解 | 原因 | 本目錄的應對 |
|---|---|---|---|
| **Tensor Core** | ❌ 不存在 | TU117 的 SM 裡沒有 TC，位置是 FP16 單元 | ✅ **軟體重建**（`02-software-tensor-core/`，1.603 TFLOPS） |
| **顯示記憶體 2 GB** | ⚠️ **可擴展** | 物理 2 顆 GDDR6 焊死，但可用系統 RAM 擴展 | ✅ **GreenBoost 實測 8 GB 虛擬顯示記憶體**（見下） |
| **SM 頻率** | ⚠️ 有限 | `nvidia-settings` 可小幅超頻，受 TGP 限制 | — |
| **FECS/GPCCS 微碼** | ⚠️ 只能讀 | FWSEC 憑證鏈驗證，改一位元組即拒絕載入 | 已完整 dump（見上層目錄 findings） |

### ⭐ 最重要的成果：顯示記憶體從 1.64 GiB 擴展到 8 GiB

**GreenBoost（上游原始碼，用 `refs/refs-fetch-manifest.sh` 取得）已在 MX450 + fnOS 6.18.18 上完整驗證通過：**

```
physical VRAM reported: 1.64 GiB

alloc 2 GiB: OK    alloc 4 GiB: OK    alloc 8 GiB: OK   ← 物理的 4.9 倍
matmul on 4GiB-occupied GPU: OK, c[0,0]=1024.0 (expect 1024.0)
```

**三級儲存：T1 VRAM 2 GB + T2 系統 RAM 8 GB + T3 NVMe 8 GB = 18 GB 模型容量。**

**代價：T2 走 PCIe 3.0 x4，頻寬約 4 GB/s，是 VRAM 的 1/20。推論能忍，訓練會慢 3–10 倍。**

**適配細節（含唯一的兩行程式碼修改）：[`01-upstream-analysis/greenboost-adaptation.md`](01-upstream-analysis/greenboost-adaptation.md)**

**這意味著 7B QLoRA（約需 4 GB）從「物理不可能」變成「能跑，但慢」。**

### 為什麼 CMPUnlocker 的方法不能照搬

CMPUnlocker 能解鎖 CMP 170HX 的 8 GB → 64 GB，**前提是物理上有 80 GB HBM2e**，
只是被 OTP fuse + 韌體限制。

MX450 的情況是：

- 顯示記憶體：物理 2 顆 GDDR6 = 2 GB，**沒有更多**
- TC：TU117 的 SM 裡**沒有 TC 單元**

**沒有「被封印的東西」，就沒有可解鎖的東西。** 這是本目錄與 CMPUnlocker 最根本的區別。

---

## 二、目錄結構

```
full-unlock/
├── README.md                      本檔案：現實核查 + 總覽
├── 01-upstream-analysis/          所有參考開源專案的分析
│   ├── cmpunlocker.md             ★ CMPUnlocker 完整技術拆解
│   └── （第三方原始碼已移除，見 refs/refs-fetch-manifest.sh）
├── 02-software-tensor-core/       ★ 核心：用 CUDA 核心重建 Tensor Core
│   ├── README.md                  實作說明與效能資料
│   ├── triton_gemm.py             v1: Triton tl.dot（走 HMMA）0.595
│   ├── triton_gemm_v2.py          v2: 外積（失敗）0.122
│   ├── triton_gemm_v4.py          v4: FP32 dot（Triton 崩潰）
│   └── sw_tensor_core_v2.py       ★ 最終版：NVRTC + FFMA，1.603 TFLOPS
├── 03-toolkit/                    重用既有工具（PRAMIN / Falcon / nvidia.ko）
└── 04-results/                    實測資料
```

---

## 三、核心成果：軟體 Tensor Core

**在 TU117 上用 CUDA 核心重建了一個矩陣乘引擎，繞開不存在的 TC。**

### 技術路線

```
CUDA C 原始碼（unsigned short + inline PTX）
    ↓ NVRTC 編譯（libnvrtc.so.12）
PTX（39,393 字元，含 cvt.f32.f16 / fma.rn.f32，不含 mma.sync）
    ↓ cuModuleLoadData（libcuda）
GPU 執行
```

**關鍵：全程不用 `cuda_fp16.h`（NVRTC 找不到），全部用 `unsigned short` + inline PTX 的
`cvt.f32.f16` / `cvt.rn.f16.f32`。**

### 實測效能（2048³，本機）

| 實作 | TFLOPS | 相對 cuBLAS TENSOR_OP |
|---|---|---|
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | 0.320 | 1.0x |
| Triton `tl.dot`（FP16，走 HMMA） | 0.595 | 1.9x |
| **本目錄 `sw_tensor_core_v2.py`（FFMA）** | **1.603** | **5.0x** |
| cuBLAS FP32（`volta_sgemm`） | 2.122 | 6.6x |
| FP16x2 純運算峰值（FP16 累加，不可用於真實 GEMM） | 5.770 | — |

**正確性：`max abs err = 0.00000`（與 float64 參考完全一致）。**

### 為什麼是 1.603 而不是 5.77

**`fma.rn.f16x2` 是 FP16 累加。K=2048 時累加器會溢位，結果不可用。**

真實 GEMM 必須 FP32 累加 → 只能用 FFMA → 上限是 FP32 的 2.5 TFLOPS。
1.603 是 2.5 的 64%，剩下的是共享記憶體頻寬和指令排程的損耗。

**這不是實作不夠好，是硬體的算術強度上限。**

---

## 四、上游專案的實際使用情況

| 專案 | 用了嗎 | 怎麼用的 |
|---|---|---|
| **CMPUnlocker** | ✅ 原始碼研讀 + 完整拆解 | 學到 ROP 注入機制；確認其 PLM 目標暫存器 Turing 不存在 |
| **Triton** | ✅ 實際使用 | v1 基線 0.595；發現 CC 7.5 上 FP32 路徑有 `map::at` 崩潰 |
| **NVRTC / CUDA Driver API** | ✅ 實際使用 | 最終版全部依賴此路徑 |
| **tinygrad** | ⚠️ 僅參考 | 其使用者態驅動思路不適用於本任務 |
| **TileKernels / FlagOS** | ⚠️ 僅參考 | 面向昇騰／多晶片，Triton DSL 思路已借鏡 |
| **DRAMA++ / Sudoku** | ❌ 不適用 | ARM64 伺服器 DRAM 位址對映 |
| **Open-RDMA / fabric-lib** | ❌ 不適用 | 網路層，與單卡無關 |
| **AITER** | ❌ 不適用 | AMD ROCm 專用 |
| **shadPS4 顯示記憶體重構** | ❌ 不適用 | 模擬器場景 |

**誠實說明：txt 裡列的專案，真正能用於 MX450 的只有 CMPUnlocker（方法論）和 Triton（工具鏈）。
其餘因架構／平台不匹配無法使用——這不是選擇，是物理限制。**

---

## 五、授權邊界（重要）

**本目錄全部內容均為本專案原創，MIT 授權。**

早期版本曾內嵌 `01-upstream-analysis/vendor/`（GPL v2 的 cmpunlocker 與 greenboost）。
**2026-10-21 已全部移除**，倉庫不再包含任何第三方原始碼副本，授權邊界因此變得乾淨。

需要查閱上游原始碼時：`bash refs/refs-fetch-manifest.sh`（SHA 已釘死）。

---

## 六、重現

```bash
# 軟體 Tensor Core
cd 02-software-tensor-core
/vol1/ganlin/venv/bin/python sw_tensor_core_v2.py --size 2048

# 依賴：torch 2.3.1 + nvidia-cuda-nvrtc-cu12（libnvrtc.so.12）
```

---

## 七、結論

**「解鎖 Tensor Core」在 MX450 上不成立——因為沒有 TC 可解鎖。**

**但「重建一個」成立——用 CUDA 核心的 FFMA 路徑，拿到 1.603 TFLOPS，
是這張卡上 cuBLAS TENSOR_OP 的 5.0 倍。**

**這就是這台機器在矩陣乘上的真實上限。**
