# 知乎兩篇 GPU 逆向方法論文章 —— 摘要歸檔與對本專案的對映

> 來源（2026-10-20 閱讀）：
> ① 《GPU逆向方法總結》 https://zhuanlan.zhihu.com/p/2014778262367250116 （編輯於 2026-03-10）
> ② 《Tensor Core 微架構深度解析》 https://zhuanlan.zhihu.com/p/2028548125518755193
> 本檔案為結構化歸檔（含全部關鍵資料表、論文／工具清單），並標註與本專案 MX450 工作的對應關係。

---

## 一、文章①《GPU逆向方法總結》—— 八類方法

| 章節 | 方法 | 手段 | 可得到什麼 |
|---|---|---|---|
| 0x01 | 微基準測試 | `clock`/`clock64` 計數 + 控制變數 | 管線延遲／吞吐、排程策略、TC 特性 |
| 0x02 | ISA 二進位分析與補丁 | bit-flipping、SASS 補丁 | 指令編碼格式、控制碼語意、依賴屏障 |
| 0x03 | 記憶體層次探測 | Pointer-Chasing | 快取大小／相聯度／替換策略、TLB 層次、延遲 |
| 0x04 | 二進位插樁 | NVBit / SASSI | 執行時指令級行為 |
| 0x05 | 模擬器驅動 | GPGPU-Sim / Accel-Sim | 假設顯式化、硬體-模型對比 |
| 0x06 | 側通道與安全 | timing / contention | 排程器粒度、共享資源邊界 |
| 0x07 | 功耗與熱建模 | 微基準 + 回歸 | 元件級功耗、功率閘控粒度 |
| 0x08 | 工具與組譯器 | CuAssembler / maxas / TuringAs / DocumentSASS | SASS 級修改與優化 |

### 1.1 關鍵論文與資料（按世代）

| 論文 | 架構 | 關鍵發現 |
|---|---|---|
| Wong 2010 (ISPASS) | GT200 | 三級指令快取 4/8/32 KB；L1 TLB 16 路全相聯對映 8MB、行 512KB；L2 TLB 32MB 8 路、行 4KB |
| Jia 2018 (arXiv:1804.06826) | Volta V100 | L2 6144 KiB / 900 GB/s @877MHz；L2 實體索引；L1.5 常數快取與 L1I 共享；**6 個依賴屏障** |
| Jia 2019 (arXiv:1903.07486) | Turing T4 | 指令編碼與 Volta 相同；**雙埠暫存器 bank**；**reuse flag 每條最多 4 操作數**；L2 16 路；**L1 miss/L2 hit=188 cycles，L2 miss/TLB hit=296，full miss=616** |
| Abdelkhalik 2022 (arXiv:2208.11174) | Ampere A100 | 全類型 CPI 目錄；**SASS barrier 約 33 週期** |
| Arafa 2019 (arXiv:1905.08778) | 5 代 7 卡 | 低開銷指令延遲特徵化框架（NMSU-PEARL/GPUs-ISA-Latencies） |
| Luo 2024 (arXiv:2402.13499) | Hopper H100 | 首次特徵化 **WGMMA**；**平均 L2 延遲 = L1 的 6.5 倍** |
| Luo 2025 (arXiv:2501.12084) | Hopper | TMA 1.5x；FP8 ≈ 2× FP16；DPX ≥4.75x |
| Huerta 2025 (arXiv:2503.20481, MICRO) | Ampere/Turing/Blackwell | **CGGTY 排程策略，無傳統記分板、無 operand collector**；每 bank 1024-bit 讀埠；RFC 每 entry 3×1024-bit；**41 控制位/warp = 暫存器堆 0.09%**；Accel-Sim MAPE 33%→13.45% |
| Jarmusch 2025a (arXiv:2512.02189) | Blackwell B200 | ResNet-50 1.85x vs H200；快取未命中延遲 −58%；解壓縮引擎 42–462 GB/s；首次特徵化 TMEM |
| Jarmusch 2025b (arXiv:2507.10789) | Blackwell GB203 | L2 命中約 358 cycles（固定）；GH100 L2 65MB |
| Mei & Chu 2017 (arXiv:1509.02308) | Fermi/Kepler/Maxwell | **GPU L1 資料快取不用 LRU** → 推翻傳統 P-chase 假設 |
| Zhang 2023 (CCS) | Turing/Ampere | **MIG 不分區 L3-uTLB**；L3-uTLB 8 路 2 slice，每 slice 8 entry victim buffer |
| Sun 2023 (TPDS) | Volta/Turing/Ampere | ThC 世代：Volta 8 個/SM 4×4×4 僅 FP16；Turing +INT8/INT4/Binary；Ampere 4 個/SM 8×4×8 +BF16/TF32/稀疏 |
| Khairy 2019/2020 | Volta / 全代 | 記憶體模型誤差 −66x；**Accel-Sim** 80 負載 1945 核心，週期誤差 −79 個百分點 |
| Villa 2019 (NVBit) / Stephenson 2015 (SASSI) | 多代 | 動態／靜態插樁框架 |
| Leng 2013 / Kandiah 2021 | GPUWattch / AccelWattch | 誤差 9.9%/13.4%；AccelWattch Volta MAPE 7.5–9.2%；**功率閘控粒度：晶片級 L2 / SM 級 L1 / lane 級 FPU** |
| Naghibijouybari 2017/2018 | Maxwell/Volta | 隱蔽通道 >4 Mbps；側通道致 **CVE-2018-6260** |

### 1.2 ★ 工具清單（可直接用於 MX450）

| 工具 | 倉庫 | 對我們的用途 |
|---|---|---|
| **CuAssembler** | `cloudcores/CuAssembler` | Kepler~Hopper 全代 SASS 反組譯／組譯／補丁 → **改我們自己 GEMM 的 stall/barrier/reuse 控制碼** |
| **TuringAs** | `daadaada/turingas` | **Volta/Turing 的 SASS 組譯器**，直接對應 sm_75 |
| **maxas** | `NervanaSystems/maxas` | Maxwell SGEMM 達理論峰值 **98%**、比 cuBLAS 快 **4.8%** → 我們 80.9%→~95% 的樣板 |
| **DocumentSASS** | `0xD0GF00D/DocumentSASS` | `LD_PRELOAD` 攔 nvdisasm，抽取**內嵌的指令規範與排程延遲** → 省掉微基準測固定延遲 |
| **nv_isa_solver** | `kuterd/nv_isa_solver` | fuzz nvdisasm 自動生成 ISA 規範（已出 SM90a/SM89） |
| **AmpItUp** | `NoxNode/AmpItUp` | Ampere 指令編碼逐位註解（方法可套 sm_75） |
| **NVBit / SASSI** | `NVlabs/NVBit` / `NVlabs/SASSI` | 插樁我們自研 kernel |
| **Accel-Sim** | `accel-sim/accel-sim-framework` | 校核微基準結論 |
| **gpu-tlb** | `0x5ec1ab/gpu-tlb` | TLB 逆向（含 pagemap/頁表工具） |
| **CuAsmRL / SIP** | arXiv:2501.08071 / 2403.16863 | 自動搜 SASS 排程，實測 **+9%（最高 26%）** |
| JEB 5.31+ | 商業 | Volta~Blackwell 反編譯 |

### 1.3 可直接照搬的方法論要點

1. **CuAsmRL 的教訓**：基於時鐘的 SASS 微基準**可能不準**（它把 IADD3 測成 2.6 cycles），
   **基於依賴的測法**（逐步降低 stall count 直到結果出錯）更可靠。→ 我們測延遲時應改用此法。
2. **控制碼是效能的隱藏層**：cuBLAS 的優勢主要來自 SASS 級管線管理，而非演算法（Schuetze 2024 逆向 cuBLAS 結論）。
3. **TuringAs 的關鍵優化**：用 8 個 FFMA 交錯 LDG（cuDNN 只用 2 個）→ 單項 **1.24x**。
   這與我們 v7 的「雙緩衝 + 交錯載入」方向一致，但它是 SASS 級精確控制。
4. **NVIDIA 專利是逆向情報源**：`US20150220341A1`（軟體記分板 DEPBAR，6-bit 計數 0–63）、
   `US11080051B2`（LDGSTS 非同步拷貝 + LDGDEPBAR）。

---

## 二、文章②《Tensor Core 微架構深度解析》—— ★ 直接解釋我們的實測

### 2.1 基礎事實

- Volta SM = 4 Sub-Core，每 Sub-Core：1 Warp Scheduler（1 instr/clk）、1 Dispatch、RF 64KB、**2 個 Tensor Core**、8 FP64、16 FP32、16 INT、4 MUFU、MIO、L0/L1 ICache
- Titan V：80 SM × 8 TC = 640 TC；1530 MHz → **理論 125 TFLOPS（FP16 混合精度）**
- PTX `wmma.mma` → 多條 **HMMA** SASS 指令

### 2.2 ★★ HMMA.884 的 SET/STEP 結構（與我們的 MX450 直接對應）

```
HMMA.884.F32.F32.STEP0 R8,  R24.reuse.COL, R22.reuse.ROW, R8;   ← 第10週期
HMMA.884.F32.F32.STEP1 R10, R24.reuse.COL, R22.reuse.ROW, R10;  ← 第12週期
HMMA.884.F32.F32.STEP2 R4,  R24.reuse.COL, R22.reuse.ROW, R4;   ← 第14週期
HMMA.884.F32.F32.STEP3 R6,  R24.COL,       R22.ROW,       R6;   ← 第18週期
```

| 事實 | 數值 |
|---|---|
| 一個 `wmma.mma` 16×16×16 混合精度 → **16 條 HMMA**（4 SET × 4 STEP） | 累積 44 週期（10→54） |
| FP16 累加模式 → **8 條 HMMA**（4 SET × 2 STEP） | 總延遲 64 週期 |
| 單個暫存器編號 = **一對相鄰暫存器** | R8 代表 R8:R9 |
| `reuse` 標註 = 快取到 **Operand Reuse Cache** | 每指令最多 4 操作數 |
| 每 SET：4×4 A × 4×8 B → 4×8 C/D；每 STEP：2×4 A × 4×4 B | 4×4=16 次基本乘加 |
| 每 TC = 16 個 FEDP 單元，4 級管線 | 暫存器頻寬 6kb/2cycle/warp = 1.5kb/cycle/TC |
| **TC 不能與 INT/FP32 co-issue** | 因為它佔滿暫存器存取埠（3 源 × 512bit = 1536 bit/cycle） |
| 並行度 | 每 SM **僅 4 warp 可並行執行 HMMA**；每 warp 用 2 個 TC |

### 2.3 Turing 的時序表（Table I，16×16×16）

| 精度 | SET1 | SET2 | SET3 | SET4 | 總延遲 |
|---|---|---|---|---|---|
| FP32 累加 | 42 | 56 | 78 | 99 | **99 週期** |
| FP16 累加 | 44 | 52 | 60 | 74 | 74 週期 |
| INT8 | 40 | 44 | 47 | 59 | 59 週期 |

**Turing 的 16×16×16 混合精度需 99 週期，比 Volta 的 54 週期更慢**（但 Turing 支援 32×8×16 / 8×32×16 / 8×8×32 更大 tile）。

### 2.4 ★★★ 對本專案的决定性推論

**我們實測**：MX450(TU117, sm_75) 上 cuBLAS fp16 走 `cutlass_70_tensorop_f16_s884gemm`，
只有 **0.3191 TFLOPS = 時鐘折算峰值的 9.9%**（FP32 FFMA 可達 99.0%）。

**文章②給出的機制解釋**：
1. `s884` 就是 **`mma.m8n8k4` → `HMMA.884`** 這條路徑，一個 wmma 需要 16 條 HMMA、99 週期；
2. 而 **TU117 沒有實體 Tensor Core**（本專案已實測確認：FP16 5.77 vs mma 0.60，慢 9.6x），
   所以每條 HMMA.884 必須由普通 FP16/FEDP 通路**逐條模擬**；
3. HMMA 還**不能與 FP32/INT co-issue**（佔用全部暫存器埠），
   → 模擬時既拿不到 TC 吞吐，又阻塞純量管線 → 只剩 ~10%。

**→ 這就是 9.9% 的完整因果鏈，不是「驅動沒優化」，而是硬體缺失 + 指令序列不匹配。**

**由此可推**：在 MX450 上任何「走 HMMA 的庫」（cuBLAS fp16 / Triton / TileLang / FlagGems）
都會撞同一堵牆（實測 Triton 0.595 / TileLang 0.538 / FlagGems 0.29-0.55x，全部印證）。
**唯一出路是我們自研的 FFMA 路徑**（當時記為 2.4872 TFLOPS；見譯者註）。

---

## 三、兩篇文章 → 本專案下一步的對映（可執行）

| 文章方法 | 我們已做的 | 可直接升級的動作 |
|---|---|---|
| 0x01 微基準 | FMA 峰值 2.4747（99.0%）、GEMM 2.4872（80.9%）、時鐘取樣 | 沿用；補**依賴式測法**（CuAsmRL 教訓） |
| 0x02 SASS 分析/補丁 | 抓 kernel 名 `cutlass_70_tensorop_f16_s884gemm` | **引入 CuAssembler / TuringAs 改我們自研 GEMM 的 stall-count / barrier / reuse** |
| 0x03 記憶體層次 | DRAMA++ / Sudoku 測 DRAM bank 對映 | 補 **Pointer-Chasing 測 MX450 L1/L2/TLB**（2GB 顯示記憶體可做） |
| 0x04 插樁 | 無 | NVBit 插樁自研 GEMM，看 stall 原因 |
| 0x05 模擬器 | 無 | Accel-Sim 校核我們的 80.9% 是否已到該配置上限 |
| 0x07 功耗/熱 | 只讀了 clocks_event_reasons | **AccelWattch 思路：用微基準推功率閘控**（我們 power.draw=N/A，只能靠間接） |
| 0x08 工具 | 無 | **DocumentSASS 抽 nvdisasm 內嵌延遲表 → 精確知道 sm_75 每條指令延遲** |

**最高優先級**：0x08 的 DocumentSASS + 0x02 的 CuAssembler/TuringAs。
理由：maxas 在 Maxwell 用這套方法做到**理論峰值 98%、超 cuBLAS 4.8%**；
我們目前 80.9%，缺的正是 **SASS 級 stall/barrier 精確控制**（不是演算法、不是 tile 尺寸）。

---

## 譯者註（2026-10-23）

本歸檔是 2026-10-20 當時所讀與所測的忠實記錄，**未改寫原文**。但後續一次可靠 benchmark
（交錯 12 輪、取中位數，`reliable_bench.py`）顯示：上文的「2.4872 TFLOPS / 80.9%」來自
**單次取最優**的測量，本機跨輪漂移可達 15%。中位數結果為：

| 實作 | 中位 TFLOPS |
|---|---|
| 自研 FFMA GEMM | **1.50** |
| cuBLAS fp32 | **2.06** |

即自研核心為 cuBLAS 的 **73%**，並非超過它。本文的微架構結論
（HMMA 模擬鏈、TC 無法 co-issue、FFMA 是唯一出路）**不受影響**——只有絕對吞吐數字需要這個修正。
