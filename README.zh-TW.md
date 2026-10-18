# MX450 (TU117) Tensor Core 逆向研究

[简体中文](README.md) | **繁體中文** | [English](README.en.md)

> 查清 NVIDIA GeForce MX450（TU117M）上 Tensor Core 被降速至理論值 5–10% 的機制，
> 並系統性地探明這台機器在韌體/驅動堆疊上的所有可存取邊界。
>
> **狀態：公開資料可達的邊界已全部探明 · 未實現解鎖 · 結論以實測數據為準**

---

## 一句話結論

**這張卡上確實存在一個被鎖住的算力空間（Tensor Core 有指令、能執行、但吞吐只有理論值的 5–10%）。
這台機器用五層機制把它封住了，本次研究逐一驗證並繞過了其中四層——第五層是 NVIDIA 從未公開的暫存器語義。**

---

## 1. 專案起點：一個反常的實測現象

| 路徑 | 實測吞吐 | 說明 |
|---|---|---|
| 手寫 PTX `mma.m16n8k8`（Tensor Core 指令） | **0.60 TFLOPS** | 走 TC 路徑 |
| 手寫 PTX `FP16x2`（CUDA 核，不碰 TC） | **5.77 TFLOPS** | 純 FP16 單元 |
| 手寫 PTX `FP32 FMA` | 3.06 TFLOPS | 基準 |
| `cuBLAS` FP16，日誌確認 `CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)` | **0.32 TFLOPS** | NVIDIA 自家的函式庫 |
| `cuBLAS` FP32（不用 TC） | 2.07 TFLOPS | 對照 |
| TU117 理論張量峰值（14 SM × 4 TC） | 6 – 12 TFLOPS | — |

**TC 指令存在、能編譯、能執行、不報錯 —— 但比不用 TC 還慢 5–6 倍。**

---

## 2. 五層封鎖與突破結果

| # | 封鎖層 | 常規認知 | 本次結果 |
|---|---|---|---|
| ① | **硬體層**：GPU 無獨立 SPI Flash | 無法刷 vBIOS | ❌ 無法突破（物理事實） |
| ② | **系統韌體**：Intel Boot Guard 簽章加密 | 無法解析 BIOS | ❌ 無法突破（需廠商私鑰） |
| ③ | **GPU 韌體**：GSP FWSEC 憑證鏈驗證 | 無法修改微碼 | ❌ 無法突破（需 NVIDIA 私鑰） |
| ④ | **驅動層**：NVML / CUPTI / 私有 ioctl 逐層裁剪 | 無法讀取底層 | ✅ **部分突破**（PRAMIN 通道） |
| ⑤ | **暫存器語義**：GPC/TPC 偏移未公開 | — | ⚠️ **未解**（資訊不存在） |

---

## 3. 已證實的 21 條硬事實

| # | 事實 | 證據 |
|---|---|---|
| 1 | TC 存在但被限速至理論值 5% | 手寫 PTX 0.60 vs 理論 6–12；cuBLAS 同現象 |
| 2 | GSP 韌體不含 TC 控制邏輯 | 雙層 ELF 反組譯，8 個 Ampere 暫存器位址零命中 |
| 3 | vBIOS 受 FWSEC 憑證鏈保護 | 韌體內 8 個 `VERIFY_*` 錯誤碼 |
| 4 | BAR1「僅映射 16MB」可繞過 | PRAMIN 通道，寫-讀回驗證通過 |
| 5 | P2P「消費級 EINVAL」可繞過 | 不走 P2P，直走 PRAMIN |
| 6 | 驅動運行時 BAR0 可存取 | 核心模組 `ioremap` |
| 7 | 視訊記憶體可讀 | 32 MB dump，MD5 可校驗 |
| 8 | 視訊記憶體不含 Falcon 微碼 | Kepler 微碼簽章零命中 |
| 9 | FECS Falcon 基底 = `0x409000` | 暫存器實測 |
| 10 | `HWCFG1` 位於 `+0x12c`（非 `+0x10c`） | 原始碼確認 |
| 11 | **FECS 執行在 LIGHT SECURE 模式** | `security_model = 2`（非 HS 黑箱） |
| 12 | **FECS 微碼可完整讀取** | 32 KB 全量落盤 |
| 13 | 微碼與 gk20a 同源結構 | 7 位元組簽章 `00fe0400bda47e` 匹配 |
| 14 | 入口 = 設定堆疊頂端 + `lcall main` + `exit` | 反組譯確認 |
| 15 | ISA 為 fuc6 | `core_rev = 6` |
| 16 | 449 條暫存器寫入，位址 100% 解析 | 靜態分析 |
| 17 | 379 次落在 `0x20000–0x2FFFF` 段 | 分段統計 |
| 18 | **該段語義未知** | 三條驗證路線均失敗 |
| 19 | PGRAPH 在 `0x400000` | envytools rnndb 權威定義 |
| 20 | 微碼不可修改 | FWSEC 簽章驗證 |
| 21 | 視訊記憶體 1.68 GB 不可變 | 物理焊死 |

---

## 4. 關鍵突破：如何讀出 FECS 微碼

五道鎖逐一打開：

| 鎖 | 常規認知 | 繞過方法 |
|---|---|---|
| BAR1 只映射 16 MB | 讀不到視訊記憶體 | 改走 **PRAMIN**（`BAR0+0x700000`） |
| P2P 對 GeForce 回傳 EINVAL | 無法 pin VRAM | 不走 P2P |
| 驅動運行時 BAR0 被佔用 | 無法存取暫存器 | **核心模組 `ioremap`** |
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
| `vram_dump.bin` — 視訊記憶體 dump | 32 MB | MD5 `1b224dcfeec507673ee64fd35d44005c` |
| `bios_readonly.bin` — 系統 BIOS | 24 MB | MD5 `4d2c99e85e761e0a3b1b8dcada4cc950` |
| `envytools/envydis` — 反組譯器 | — | 自行編譯 |

> 二進位產物因體積未納入倉庫，可透過 `tools/` 下的腳本重新產生。

---

## 6. 倉庫結構

```
.
├── README.md                    简体中文
├── README.zh-TW.md              繁體中文（本檔案）
├── README.en.md                 English
├── LICENSE
├── docs/
│   ├── 02-技术路线图.md
│   └── 03-内核模块路线图.md
├── findings/                    二十三篇實測記錄（append-only）
├── refs/                        NVIDIA 官方標頭檔（MIT，原樣引用）
├── kernel/                      6 個核心模組
│   ├── mx450_probe.c            工具鏈驗證
│   ├── mx450_p1.c               kprobe 符號解析
│   ├── mx450_p2.c               核心記憶體 no-fault 讀取
│   ├── mx450_p3.c               PRAMIN 視訊記憶體存取
│   └── mx450_p4.c               FECS IMEM 讀取
├── tools/                       20 個分析工具
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
| 視訊記憶體 | 2048 MiB GDDR6，64-bit，單次配置上限 1620 MiB |
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

**`0x20000` 段的語義無法確定**，三條路線全部嘗試：

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
4. **視訊記憶體 1.68 GB 是物理焊接**，任何軟體手段都無法改變。
5. **僅供研究與教育用途**，僅適用於你自己合法擁有的硬體。

---

## 11. 聲明

本專案與 NVIDIA 無任何關聯。所有商標歸其各自所有者。
倉庫內不含 NVIDIA 專有程式碼或韌體；`refs/` 下均為 NVIDIA 以 MIT 授權公開發布的原始標頭檔。

## 授權

MIT（見 `LICENSE`）。研究成果歡迎引用，使用風險自負。
