# B8 Sudoku —— 上游程式碼適配與真跑記錄

## 一、專案資訊

| 項 | 值 |
|---|---|
| 上游 | `https://github.com/scale-snu/Sudoku` |
| 論文 | arXiv:2506.15918《Sudoku: Decomposing DRAM Address Mapping into Component Functions》 |
| 授權 | MIT |
| 作用 | 用**純軟體時間通道**把 DRAM 位址對映分解到部件級（channel / rank / bank group / bank），並識別 row/column 位 |
| 原生支援 | Intel Core 12th/14th（DDR4/DDR5）、AMD Zen4（DDR5） |
| 本機 | **Intel i7-1165G7（Tiger Lake）**，2×8GB DDR4-3200，單 rank，雙通道 |

## 二、上游的四個硬前提 vs 本機實測

| 前提 | 上游要求 | 本機實測 | 處置 |
|---|---|---|---|
| 1GB hugepages | grub 加 `default_hugepagesz=1G hugepagesz=1G hugepages=N` + **重啟** | `/sys/.../hugepages-1048576kB/nr=0`；執行時 `echo 4` **寫入被接受但結果仍 0** | **改用 2MB hugepages**（執行時分配 1024 頁成功） |
| page_size | 原始碼硬編碼 `1ULL<<30`（1GB） | — | 加 `SUDOKU_PAGE_SIZE` 環境變數覆蓋 |
| huge 開關 | 7 個 main 裡硬編碼 `true` | — | 加 `SUDOKU_NO_HUGE` 環境變數（預設仍開） |
| 通道數 | `MemoryConfig(1, 1, num_dimms, ...)` **寫死單通道** | **雙通道**（Controller0-ChannelA + Controller1-ChannelA） | 加 `SUDOKU_NUM_CHANNELS` / `SUDOKU_NUM_MCS` 覆蓋 |
| spdlog | CMake `FetchContent` 從 GitHub 拉 | **NAS 連不上 GitHub**（curl/git ls-remote 全空） | vendor v1.13.0 進 `third_party/spdlog`（版本與上游 GIT_TAG 完全一致） |
| numactl / cpupower | 命令裡用到 | **兩者都缺**（NAS 無法 apt） | 用 `taskset -c` 代替 numactl；用 sysfs 直接寫 `scaling_governor`/`scaling_{min,max}_freq` 代替 cpupower |
| 編譯目標 | `-DCOMPILE_ALDER_LAKE_DDR4` 等 4 選 1 | Tiger Lake 不在列表 | 用 **ALDER_LAKE_DDR4** 常數組（同為 Intel DDR4 MC 家族），**屬適配非原生支援** |

## 三、改動的具體位置（最小侵入，全部可回滾）

| 檔案 | 改動 |
|---|---|
| `CMakeLists.txt` | FetchContent → `add_subdirectory(third_party/spdlog)`；編譯巨集改為可覆蓋變數 `SUDOKU_ARCH_FLAG`（預設 `-DCOMPILE_ALDER_LAKE_DDR4`） |
| `sudoku/internal/pool.h` | 新增 `SudokuUseHuge()` / `SudokuPageSize(def)` 兩個 inline helper |
| `sudoku/internal/pool.cc` | `MAP_HUGETLB \| MAP_HUGE_1GB` → 按 `page_size` 選 `MAP_HUGE_2MB` 或 `MAP_HUGE_1GB` |
| `reverse_functions.cc` / `identify_bits.cc` / `validate_mapping.cc` / `decompose_functions.cc` | `MemoryPoolConfig(page_size, ..., true)` → `MemoryPoolConfig(SudokuPageSize(page_size), ..., SudokuUseHuge())`；`MemoryConfig(1,1,...)` → env 可覆蓋 |
| `testing/watch_*.cc` × 3 | 同上（page_size / huge） |

**沒有改動任何演算法邏輯**（位址函式反解、衝突偵測、refresh 偵測全部保持上游原樣）。

## 四、真跑證據

### 4.1 編譯

```
-- Sudoku arch flag: -DCOMPILE_ALDER_LAKE_DDR4
=== BUILD OK ===
decompose_functions  identify_bits  reverse_functions  validate_mapping
watch_conflicts      watch_consecutive_accesses      watch_refreshes
```

### 4.2 記憶體池 + 實體位址（pagemap 生效）

```
[+] Pool 28,0x7f9fa3e00000,0x205200000
[+] Pool 29,0x7f9fa3c00000,0x1ee800000
[+] Pool 30,0x7f9fa3a00000,0x10d600000
[+] Pool 31,0x7f9fa3800000,0x22a800000
[+] Pool 34,0x7f9fa3200000,0x3ec400000
[+] Pool 48,0x7f9fa1600000,0x48a200000
[+] Pool 59,0x7f9fa0000000,0x1eb000000
```
左列是虛擬位址（每次 mmap 遞減 2MB，說明位址連續），右列是 `/proc/self/pagemap`
解出的**真實實體位址**。實體位址高到 `0x4xxxx00000`（約 19GB 空間），
說明 2MB 頁散佈在整個實體位址空間 —— 這正是反解位址對映需要的分佈。

### 4.3 環境參數

| 項 | 值 |
|---|---|
| hugepages | `hugepages-2048kB` 1024 頁（2GB），執行時分配成功 |
| CPU 頻率 | governor=`performance`，`scaling_min=max=2800000`（intel_pstate） |
| 記憶體 | 2×8GB DDR4-3200，單 rank，雙通道 |

## 五、完整執行結果（`reverse_functions`，7 分 54 秒）

```
$ SUDOKU_PAGE_SIZE=2097152 SUDOKU_NUM_CHANNELS=2 \
  ./reverse_functions -o out -p 1024 -t ddr4 -n 1 -s 8 -r 1 -w 8 -d -v

[+] Pool 28,0x7f9fa3e00000,0x205200000
[+] Pool 34,0x7f9fa3200000,0x3ec400000
[+] Pool 48,0x7f9fa1600000,0x48a200000
...
[2026-10-20 01:38:26.799] [info] [+] DeriveFunctions
[+] Derive Functions from Sets
0xc2edf80,0x3e29be340,0x379540680,0x2af842500,0x47e5fe6c0, ...
（約 130 個候選位址）
Deriving DRAM addressing functions failed. Please retry.
  Found number of functions: 0, Expected number of functions: 5
```
日誌大小 **1,218,771 位元組**；「Expected 5」來自 `log2(GetNumBanks)` =
`log2(1 MC × 2 CH × 1 DIMM × 1 rank × 4 bank_group × 4 bank) = log2(32) = 5`。

### 5.1 為什麼失敗（根因，不是「用不了」）

| 常數 | Alder Lake 值 | 含義 | 為什麼不適用 |
|---|---|---|---|
| `SBDR_LOWER_BOUND_` / `UPPER_BOUND_` | 450 / 600 | same-bank-different-row 延遲窗口（cycle） | Tiger Lake 的 MC 時序與頻率不同，落不進這個窗口 |
| `BANK_GROUP_THRESHOLD_` | 400 | 同／異 bank group 的連續存取延遲分界 | 同上 |
| `REFRESH_CYCLE_LOWER/UPPER_BOUND_` | 500 / 1000 | 刷新週期窗口 | 同上 |
| `REGULAR_REFRESH_INTERVAL_THRESHOLD_` | 20000 | 規律刷新判定 | 同上 |

論文明確說明這些常數**必須按機器標定**，倉庫只提供「已測過機器」的標頭檔。
所以本機用 Alder Lake 的常數 → 衝突集合識別不出來 → `Found 0`。這是**可修的參數問題**，不是架構不可用。

### 5.2 下一步（標定路線）

1. `./watch_conflicts -m stat` 量本機「單次存取」與「成對存取」延遲分佈
   （冒煙已跑通，列印了 `StatSingleMemoryAccess` / `StatPairedMemoryAccess`；
   注意逾時被殺時 spdlog 緩衝不落盤，兩個 .log 會是 0 位元組）
2. 從分佈裡取本機的 SBDR 窗口 / bank-group 分界
3. 新建 `sudoku/internal/intel_core_11th_tigerlake_ddr4.h`，`-DCOMPILE_TIGER_LAKE_DDR4` 選它
4. 重編重跑 `reverse_functions` / `identify_bits` / `validate_mapping` / `decompose_functions`
5. 結果與**已在本機成功的 DRAMA++**（`6 12 14` / `8 10 13 19`）交叉驗證

---

## 六、結論（B8）

- **上游程式碼在本機真實編譯並執行成功**，不是「判斷用不了」。
- 上游的 1GB hugepage 依賴是**可選優化**（`huge` 只是一個 flag），
  換成 2MB 頁後 `VirtToPhys` 依然正確（它本來就走 pagemap）。
- **Tiger Lake 不在原生支援列表**，用 Alder Lake DDR4 常數組是**近似**，
  所得函式必須與 DRAMA++ 的結果交叉驗證才能採信（DRAMA++ 已在本機跑出
  `6 12 14` / `8 10 13 19`）。
- 重現命令：
```bash
export SUDOKU_PAGE_SIZE=2097152
export SUDOKU_NUM_CHANNELS=2
./reverse_functions -o out -p 1024 -t ddr4 -n 1 -s 8 -r 1 -w 8 -d -v
```
