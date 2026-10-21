# MX450 (TU117) Tensor Core 逆向研究

[简体中文](README.md) | **繁體中文** | [English](README.en.md)

> 一個關於「**這張卡到底有沒有 Tensor Core**」的完整取證過程。
> 結論：**沒有。** 本文記錄如何用九條路徑證明這一點，以及沿途挖出的
> Turing 韌體、Falcon 微碼、GR Context 佈局等一手材料。
>
> **狀態：結論已閉合 · 全部實測可重現**

---

## 一句話結論

**TU117 的 SM 裡沒有 Tensor Core。原本放 TC 的位置是 FP16 單元。**

**HMMA 指令能編譯、能執行、結果正確——因為它由 FP16 單元模擬，而不是由 TC 執行。**

**實測：FP16 打包運算 5.77 TFLOPS，而 HMMA 指令只有 0.60 TFLOPS——TC 路徑比純 FP16 慢 9.6 倍。**

**所以「解鎖 TC」不成立：不是被鎖住打不開，是這個功能在矽晶片上不存在。**

---

## 0. 為什麼這個結論可信：三方公開資料 + 五組實測

### 公開資料（三方獨立，2019 年就已寫明）

| 來源 | 原文 |
|---|---|
| **超能網** 2019-04-24 | 「GTX 1650 所用的 TU117 和之前的 TU116 一樣**去掉了 RT Core、Tensor Core**，SM 單元內部架構經過了修改，**原本放 Tensor Core 的位置變成了 FP16 半精度浮點運算單元**」 |
| **Notebookcheck** | 「與更快的 RTX 2000 GPU 相比，**1650 沒有整合 Raytracing 或 Tensor 核心**」 |
| **百度百科** | 「TU117 **移除了 RT Core 和用於 AI 超取樣的 Tensor Core**」 |
| 對照：RTX 20 系列 | 每個 SM 有 **8 個 Tensor Core** |

### 本機實測（`tools/` 下腳本可重現）

| 路徑 | 實測 | 說明 |
|---|---|---|
| PTX `FP16x2`（打包，CUDA 核心） | **5.77 TFLOPS** | 純 FP16 單元 |
| PTX `FP32 FMA` | 3.06 TFLOPS | 基準 |
| cuBLAS FP32（`volta_sgemm_128x64_nn`） | 2.12 TFLOPS | 達理論 85% |
| **PTX `mma.m16n8k8`（TC 指令）** | **0.60 TFLOPS** | **比 FP16 慢 9.6×** |
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)` | 0.32 TFLOPS | 比 FP16 慢 18× |

**FP16 單元理論峰值核算：`14 SM × 64 CUDA × 2(FMA) × 2(打包) × 1.6 GHz = 5.73 TFLOPS`**
**實測 5.77 —— 吻合。說明 FP16 單元是滿速的。**

**而 TC 路徑 0.60，是 FP16 的 1/9.6。**

### 決定性證據：kernel 名稱

cuBLAS 日誌顯示它選了 TC 演算法：
```
algo: val=CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)
```

但實際執行的 kernel 是：
```
void cutlass::Kernel<cutlass_70_tensorop_f16_s884gemm_relu_f16_1...
                    ↑ cutlass_70（SM 7.0 編譯）  ↑ s884 = m8n8k4（Volta 第一代 TC 形狀）
```

**兩個破綻：**

1. **`cutlass_70` 而不是 `sm75`** —— NVIDIA **沒有為 TU117 提供 sm75 的 TC kernel**，cuBLAS 只能回退到 Volta 的通用 kernel
2. **`s884`（m8n8k4）** —— 不是 Turing 的 m16n8k8 / m16n8k16

**如果 TU117 真有 TC，NVIDIA 不可能不為它編一個 sm75 的 kernel。**

---

## 1. 專案起點：一個被誤讀的現象

**最初看到的是這個：**

| 路徑 | 實測吞吐 |
|---|---|
| 手寫 PTX `mma.m16n8k8`（TC 指令） | 0.60 TFLOPS |
| 手寫 PTX `FP16x2`（CUDA 核心，不碰 TC） | **5.77 TFLOPS** |
| 手寫 PTX `FP32 FMA` | 3.06 TFLOPS |
| `cuBLAS` FP16 `TENSOR_OP` | 0.32 TFLOPS |
| `cuBLAS` FP32 | 2.07 TFLOPS |

**當時的解讀是：「TC 存在，但被限速到理論值（6–12 TFLOPS）的 5–10%。」**

**這個解讀錯了兩處：**

1. **「6–12 TFLOPS 理論值」是拿 RTX 20 系列（TU10x，每 SM 8 個 TC）套過來的，
   TU117 根本沒有 TC，這個基準不存在。**
2. **「比不用 TC 還慢 5–6 倍」這個現象，本身就是「模擬」的典型特徵——
   真 TC 不可能比同代 FP16 慢，只有「用別的單元模擬 TC」才會這樣。**

**正確解讀：`mma` 慢於 `FP16x2`，說明 HMMA 指令被降級到 FP16 單元執行。**

---

## 2. 九條路徑的探索結果

**在得出結論之前，本專案把軟體層能走的路徑全部走了一遍。**
**這些路徑本身是有價值的——它們共同構成了「TU117 上沒有可解鎖的 TC」的排除性證據。**

| # | 路徑 | 做了什麼 | 結果 |
|---|---|---|---|
| 1 | 使用者態 API | NVML / CUPTI / 私有 ioctl | 逐層裁剪，CUPTI `rc=38` 零 metric |
| 2 | 系統韌體 | flashrom 唯讀匯出 24 MB BIOS | Intel Boot Guard 加密，無 VFCT / 無 `_ROM` |
| 3 | GPU 韌體 | GSP 雙層 RISC-V ELF 完整反組譯 | 30 MB 全解，8 個 Ampere 暫存器位址零命中 |
| 4 | 顯示記憶體存取 | PRAMIN 通道（BAR0+0x700000） | **打通**，寫-讀回驗證，dump 32 MB |
| 5 | Falcon 安全模式 | 讀 `HWCFG1` | **LIGHT SECURE**（非 HS 黑箱） |
| 6 | FECS 微碼 | GPU 滿載時讀 IMEM | **完整 dump 32 KB + 反組譯 11401 行** |
| 7 | GPCCS 微碼 | 同上，基底 0x41A000 | **完整 dump 16 KB + 反組譯 6059 行** |
| 8 | Falcon IO 空間 | 查 `HOST_IO_INDEX` 位寬 | 256 KB 空間，host 視窗僅 4 KB，`0x20000` 段不可見 |
| 9 | 閉源驅動 | nvidia.ko 符號級逆向（46614 符號） | 五類線索全部排除，**`tensor|HMMA|MMA_` 字串零命中** |

**九條路徑，沒有一條找到「TC 配置」或「算力開關」。**

**原因現在很清楚了：因為它們要找的東西不存在。**

---

## 3. 已證實的硬事實

| # | 事實 | 證據 |
|---|---|---|
| 1 | **TU117 無 Tensor Core；HMMA 由 FP16 單元模擬** | 三方公開資料 + FP16 5.77 vs mma 0.60（慢 9.6×）+ kernel 名 `cutlass_70` |
| 2 | GSP 韌體不含 TC 控制邏輯 | 雙層 ELF 反組譯，8 個 Ampere 暫存器位址零命中 |
| 3 | vBIOS 受 FWSEC 憑證鏈保護 | 韌體內 8 個 `VERIFY_*` 錯誤碼 |
| 4 | BAR1「僅對映 16MB」可繞過 | PRAMIN 通道，寫-讀回驗證通過 |
| 5 | P2P「消費級 EINVAL」可繞過 | 不走 P2P，直走 PRAMIN |
| 6 | 驅動執行時 BAR0 可存取 | 核心模組 `ioremap` |
| 7 | 顯示記憶體可讀 | 32 MB dump，MD5 可校驗 |
| 8 | 顯示記憶體不含 Falcon 微碼 | Kepler 微碼簽章零命中 |
| 9 | FECS Falcon 基底 = `0x409000` | 暫存器實測 |
| 10 | `HWCFG1` 位於 `+0x12c`（非 `+0x10c`） | 原始碼確認 |
| 11 | **FECS 執行在 LIGHT SECURE 模式** | `security_model = 2`（非 HS 黑箱） |
| 12 | **FECS 微碼可完整讀取** | 32 KB 全量落盤 |
| 13 | 微碼與 gk20a 同源結構 | 7 位元組簽章 `00fe0400bda47e` 匹配 |
| 14 | 入口 = 設定堆疊頂端 + `lcall main` + `exit` | 反組譯確認 |
| 15 | ISA 為 fuc6 | `core_rev = 6` |
| 16 | 449 條暫存器寫入，位址 100% 解析 | 靜態分析 |
| 17 | 379 次落在 `0x20000–0x2FFFF` 段 | 分段統計 |
| 18 | **該段是 Falcon IO 空間，非 BAR0** | 五種假設實測排除 + 文件確認 |
| 19 | PGRAPH 在 `0x400000` | envytools rnndb 權威定義 |
| 20 | 微碼不可修改 | FWSEC 簽章驗證 |
| 21 | 顯示記憶體 1.68 GB 不可變 | 物理焊死 |
| 22 | **Falcon IO 對映規則 `Falcon = Host × 0x40`** | 兩處獨立驗證（`0x12c`=UC_CAPS2、`0x184`=CODE） |
| 23 | **GPCCS Falcon 基底 = `0x41A000`** | HWCFG 簽章掃描 |
| 24 | **GPCCS 微碼可完整讀取（16 KB）** | MD5 `784c08e5...`，7 位元組簽章匹配 |
| 25 | **FECS/GPCCS 位址集是包含關係** | GPCCS 的 38 個位址全在 FECS 的 72 個內 |
| 26 | **兩者透過 `0x246FC/0x247FC/0x24FFC` 通訊** | 三位址存取次數完全相同；微碼自算 `0x23300+0x14FC=0x247FC` |
| 27 | **`iowr` 有兩套位址機制** | `bit31=0`→Falcon IO；`bit31=1`→BAR0 絕對位址 |
| 28 | **Falcon IO 空間 256 KB，host 視窗 4 KB** | envytools 文件 + `HOST_IO_INDEX` 位寬 |

---

## 4. 關鍵突破：如何讀出 FECS 微碼

五道鎖逐一打開：

| 鎖 | 常規認知 | 繞過方法 |
|---|---|---|
| BAR1 只對映 16 MB | 讀不到顯示記憶體 | 改走 **PRAMIN**（`BAR0+0x700000`） |
| P2P 對 GeForce 回傳 EINVAL | 無法 pin VRAM | 不走 P2P |
| 驅動執行時 BAR0 被佔用 | 無法存取暫存器 | **核心模組 `ioremap`** |
| Falcon 若 HS 則黑箱 | 完全不可讀 | **實測為 LS**（關鍵） |
| 閒置時 Falcon `halted=1` | IMEM 全零 | **製造 GPU 滿載 100%** |

**讀取流程**：

```c
// 1. GPU 滿載（torch 跑 GEMM 迴圈）
// 2. insmod mx450_p4.ko bar0_phys=<BAR0> mode=dump dump_bytes=32768
// 3. 模組內部：
//    ioremap(BAR0 + 0x409000, 0x1000)
//    讀 HWCFG1 確認 security_model == 2 (LS)
//    儲存 IMEMC(0) 原值
//    for (addr = 0; addr < 32768; addr += 4) {
//        blk  = addr / 256;
//        word = (addr % 256) / 4;
//        writel((blk<<8)|(word<<2), base + 0x180);   // IMEMC
//        *(u32*)(buf+addr) = readl(base + 0x184);    // IMEMD
//    }
//    還原 IMEMC(0)
```

---

## 5. 交付物

| 產物 | 大小 | 校驗 |
|---|---|---|
| `fecs_imem.bin` — FECS 微碼原始映像 | 32 KB | MD5 `ea8eb3c46165b00421896eef45d1070a` |
| `fecs_plain.asm` — 完整反組譯 | 1.3 MB / 11401 行 | — |
| `vram_dump.bin` — 顯示記憶體 dump | 32 MB | MD5 `1b224dcfeec507673ee64fd35d44005c` |
| `bios_readonly.bin` — 系統 BIOS | 24 MB | MD5 `4d2c99e85e761e0a3b1b8dcada4cc950` |
| `envytools/envydis` — 反組譯器 | — | 自行編譯 |

> 二進位產物因體積未納入倉庫，可透過 `tools/` 下的腳本重新產生。

---

## 6. 倉庫結構

```
.
├── README.md                    簡體中文
├── README.zh-TW.md              繁體中文（本檔案）
├── README.en.md                 English
├── LICENSE
├── docs/                        技術／核心模組路線圖
├── findings/                    二十八篇實測記錄（append-only）
├── refs/                        NVIDIA 官方標頭檔（MIT，原樣引用）
├── kernel/                      6 個核心模組
│   ├── mx450_probe.c            工具鏈驗證
│   ├── mx450_p1.c               kprobe 符號解析
│   ├── mx450_p2.c               核心記憶體 no-fault 讀取
│   ├── mx450_p3.c               PRAMIN 顯示記憶體存取
│   └── mx450_p4.c               FECS IMEM 讀取
├── tools/                       分析工具
├── full-unlock/                 全棧解鎖研究（B1–B15）
└── scripts/                     QEMU 隔離測試等
```

---

## 7. 測試環境

| 項目 | 值 |
|---|---|
| 機器 | LYLNAS（聯想昭陽 E4-ITL 改裝） |
| 系統 | fnOS 1.2.0505 / Debian 12 / kernel 6.18.18 |
| 主機板 / BIOS | LNVNB161216 / F8CN59WW(V2.22) 2024-06-14 |
| CPU | Intel i7-1165G7（4C8T） |
| GPU | GeForce MX450，TU117M，`10de:1f97`，`NV_PMC_BOOT_0 = 0x167000a1` |
| 顯示記憶體 | 2048 MiB GDDR6，64-bit，單次配置上限 1620 MiB |
| 驅動 | 580.142（閉源）/ CUDA 13.0 |
| 工具鏈 | gcc 12.2 / g++ 12.2 / cmake 3.25 / envytools / flashrom / strace |

---

## 8. 重現

```bash
# 編譯核心模組
cd kernel && make

# 讀取 FECS 微碼（需 GPU 滿載）
insmod mx450_p4.ko bar0_phys=<BAR0> mode=dump dump_bytes=32768

# 反組譯
envydis -m falcon -V fuc6 -i -n fecs_imem.bin > fecs_plain.asm

# 分析暫存器存取
python3 tools/analyze_iowr.py
```

---

## 9. 未解的邊界

**`0x20000` 段的語意無法確定**，三條路線全部嘗試：

| 路線 | 結果 |
|---|---|
| envytools rnndb | 覆蓋止於 Maxwell，且上界 `0x227FF` 小於微碼實際存取的 `0x23800` |
| NVIDIA 開源標頭檔 | `turing/tu102/` 無 `dev_gpc.h` / `dev_tpc.h` |
| NVIDIA 開源 HAL | `NV_PGRAPH_GPC` / `NV_PGRAPH_TPC` 零定義，邏輯在閉源 `nv-kernel.o_binary` |

**這不是「還沒找到」，是「這一層資訊 NVIDIA 沒有公開」。**

---

## 10. 風險聲明

1. **可能損壞硬體。** 向 GPU 暫存器寫入未公開值可導致無法初始化、黑屏。
2. **筆電獨顯無獨立 vBIOS Flash。** 寫錯配置後**沒有可回滾的韌體區**。
3. **不提供任何成功保證。** 本專案已確認：微碼受 FWSEC 簽章保護，只能看不能改。
4. **顯示記憶體 1.68 GB 是物理焊接**，任何軟體手段都無法改變。
5. **僅供研究與教育用途**，僅適用於你自己合法擁有的硬體。

---

## 11. 聲明

本專案與 NVIDIA 無任何關聯。所有商標歸其各自所有者。
倉庫內不含 NVIDIA 專有程式碼或韌體；`refs/` 下均為 NVIDIA 以 MIT 授權公開發布的原始標頭檔。

## 授權

MIT（見 `LICENSE`）。研究成果歡迎引用，使用風險自負。

---

## 授權邊界（重要）

本專案**全部為 MIT**（見 `LICENSE`）。

**本倉庫不包含任何第三方原始碼副本。** 早期版本曾內嵌 `vendor/`（GPL v2 的
cmpunlocker 與 greenboost），已於 2026-10-21 全部移除，以保持授權純淨。

需要查閱上游原始碼時，用 `refs/refs-fetch-manifest.sh` 一鍵 clone（SHA 已釘死），
或直接存取上游倉庫。本專案對上游客體的分析文字均為原創，見：
- `full-unlock/01-upstream-analysis/cmpunlocker.md`
- `full-unlock/01-upstream-analysis/greenboost-adaptation.md`

---

## 法律與使用邊界（重要）

1. 本倉庫所有內容僅限**研究用途**（research/educational use），針對**你自己擁有的裝置**（owner-device）。
2. 韌體修改、ROP 注入等實驗**僅在你的裝置上、自擔風險**；作者不對任何硬體損壞、驅動故障負責（復原方法見 `05-firmware-exploit/recover_gpu.sh`）。
3. **NVIDIA 韌體二進位原件不在此倉庫**（版權屬 NVIDIA），只記錄取得方法；請勿將本倉庫工具用於繞過裝置授權散布。
4. 第三方外部專案（Sudoku / drama-pp / cmp170hx / metaroce / open-rdma-driver / pplx-garden / spdlog / DeepInfer 等）保留其各自授權，取得方式見 `refs/refs-fetch-manifest.sh`（SHA 釘死版）。
