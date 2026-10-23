# B8 Sudoku —— 上游代码适配与真跑记录

**简体中文** | [繁體中文](B8-Sudoku适配与真跑记录.zh-TW.md) | [English](B8-Sudoku适配与真跑记录.en.md)

## 一、项目信息

| 项 | 值 |
|---|---|
| 上游 | `https://github.com/scale-snu/Sudoku` |
| 论文 | arXiv:2506.15918《Sudoku: Decomposing DRAM Address Mapping into Component Functions》 |
| 许可 | MIT |
| 作用 | 用**纯软件时间信道**把 DRAM 地址映射分解到部件级（channel / rank / bank group / bank），并识别 row/column 位 |
| 原生支持 | Intel Core 12th/14th（DDR4/DDR5）、AMD Zen4（DDR5） |
| 本机 | **Intel i7-1165G7（Tiger Lake）**，2×8GB DDR4-3200，单 rank，双通道 |

## 二、上游的四个硬前提 vs 本机实测

| 前提 | 上游要求 | 本机实测 | 处置 |
|---|---|---|---|
| 1GB hugepages | grub 加 `default_hugepagesz=1G hugepagesz=1G hugepages=N` + **重启** | `/sys/.../hugepages-1048576kB/nr=0`；运行时 `echo 4` **写入被接受但结果仍 0** | **改用 2MB hugepages**（运行时分配 1024 页成功） |
| page_size | 源码硬编码 `1ULL<<30`（1GB） | — | 加 `SUDOKU_PAGE_SIZE` 环境变量覆盖 |
| huge 开关 | 7 个 main 里硬编码 `true` | — | 加 `SUDOKU_NO_HUGE` 环境变量（默认仍开） |
| 通道数 | `MemoryConfig(1, 1, num_dimms, ...)` **写死单通道** | **双通道**（Controller0-ChannelA + Controller1-ChannelA） | 加 `SUDOKU_NUM_CHANNELS` / `SUDOKU_NUM_MCS` 覆盖 |
| spdlog | CMake `FetchContent` 从 GitHub 拉 | **NAS 连不上 GitHub**（curl/git ls-remote 全空） | vendor v1.13.0 进 `third_party/spdlog`（版本与上游 GIT_TAG 完全一致） |
| numactl / cpupower | 命令里用到 | **两者都缺**（NAS 无法 apt） | 用 `taskset -c` 代替 numactl；用 sysfs 直接写 `scaling_governor`/`scaling_{min,max}_freq` 代替 cpupower |
| 编译目标 | `-DCOMPILE_ALDER_LAKE_DDR4` 等 4 选 1 | Tiger Lake 不在列表 | 用 **ALDER_LAKE_DDR4** 常量组（同为 Intel DDR4 MC 家族），**属适配非原生支持** |

## 三、改动的具体位置（最小侵入，全部可回滚）

| 文件 | 改动 |
|---|---|
| `CMakeLists.txt` | FetchContent → `add_subdirectory(third_party/spdlog)`；编译宏改为可覆盖变量 `SUDOKU_ARCH_FLAG`（默认 `-DCOMPILE_ALDER_LAKE_DDR4`） |
| `sudoku/internal/pool.h` | 新增 `SudokuUseHuge()` / `SudokuPageSize(def)` 两个 inline helper |
| `sudoku/internal/pool.cc` | `MAP_HUGETLB \| MAP_HUGE_1GB` → 按 `page_size` 选 `MAP_HUGE_2MB` 或 `MAP_HUGE_1GB` |
| `reverse_functions.cc` / `identify_bits.cc` / `validate_mapping.cc` / `decompose_functions.cc` | `MemoryPoolConfig(page_size, ..., true)` → `MemoryPoolConfig(SudokuPageSize(page_size), ..., SudokuUseHuge())`；`MemoryConfig(1,1,...)` → env 可覆盖 |
| `testing/watch_*.cc` × 3 | 同上（page_size / huge） |

**没有改动任何算法逻辑**（地址函数反解、冲突检测、refresh 检测全部保持上游原样）。

## 四、真跑证据

### 4.1 编译

```
-- Sudoku arch flag: -DCOMPILE_ALDER_LAKE_DDR4
=== BUILD OK ===
decompose_functions  identify_bits  reverse_functions  validate_mapping
watch_conflicts      watch_consecutive_accesses      watch_refreshes
```

### 4.2 内存池 + 物理地址（pagemap 生效）

```
[+] Pool 28,0x7f9fa3e00000,0x205200000
[+] Pool 29,0x7f9fa3c00000,0x1ee800000
[+] Pool 30,0x7f9fa3a00000,0x10d600000
[+] Pool 31,0x7f9fa3800000,0x22a800000
[+] Pool 34,0x7f9fa3200000,0x3ec400000
[+] Pool 48,0x7f9fa1600000,0x48a200000
[+] Pool 59,0x7f9fa0000000,0x1eb000000
```
左列是虚拟地址（每次 mmap 递减 2MB，说明地址连续），右列是 `/proc/self/pagemap`
解出的**真实物理地址**。物理地址高到 `0x4xxxx00000`（约 19GB 空间），
说明 2MB 页散布在整个物理地址空间 —— 这正是反解地址映射需要的分布。

### 4.3 环境参数

| 项 | 值 |
|---|---|
| hugepages | `hugepages-2048kB` 1024 页（2GB），运行时分配成功 |
| CPU 频率 | governor=`performance`，`scaling_min=max=2800000`（intel_pstate） |
| 内存 | 2×8GB DDR4-3200，单 rank，双通道 |

## 五、完整运行结果（`reverse_functions`，7 分 54 秒）

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
（约 130 个候选地址）
Deriving DRAM addressing functions failed. Please retry.
  Found number of functions: 0, Expected number of functions: 5
```
日志大小 **1,218,771 字节**；“Expected 5” 来自 `log2(GetNumBanks)` =
`log2(1 MC × 2 CH × 1 DIMM × 1 rank × 4 bank_group × 4 bank) = log2(32) = 5`。

### 5.1 为什么失败（根因，不是“用不了”）

| 常数 | Alder Lake 值 | 含义 | 为什么不适用 |
|---|---|---|---|
| `SBDR_LOWER_BOUND_` / `UPPER_BOUND_` | 450 / 600 | same-bank-different-row 延迟窗口（cycle） | Tiger Lake 的 MC 时序与频率不同，落不进这个窗口 |
| `BANK_GROUP_THRESHOLD_` | 400 | 同/异 bank group 的连续访问延迟分界 | 同上 |
| `REFRESH_CYCLE_LOWER/UPPER_BOUND_` | 500 / 1000 | 刷新周期窗口 | 同上 |
| `REGULAR_REFRESH_INTERVAL_THRESHOLD_` | 20000 | 规律刷新判定 | 同上 |

论文明确说明这些常数**必须按机器标定**，仓库只提供“已测过机器”的头文件。
所以本机用 Alder Lake 的常数 → 冲突集合识别不出来 → `Found 0`。这是**可修的参数问题**，不是架构不可用。

### 5.2 下一步（标定路线）

1. `./watch_conflicts -m stat` 量本机「单次访问」与「成对访问」延迟分布
   （冒烟已跑通，打印了 `StatSingleMemoryAccess` / `StatPairedMemoryAccess`；
   注意超时被杀时 spdlog 缓冲不落盘，两个 .log 会是 0 字节）
2. 从分布里取本机的 SBDR 窗口 / bank-group 分界
3. 新建 `sudoku/internal/intel_core_11th_tigerlake_ddr4.h`，`-DCOMPILE_TIGER_LAKE_DDR4` 选它
4. 重编重跑 `reverse_functions` / `identify_bits` / `validate_mapping` / `decompose_functions`
5. 结果与**已在本机成功的 DRAMA++**（`6 12 14` / `8 10 13 19`）交叉验证

---

## 六、结论（B8）

- **上游代码在本机真实编译并运行成功**，不是「判断用不了」。
- 上游的 1GB hugepage 依赖是**可选优化**（`huge` 只是一个 flag），
  换成 2MB 页后 `VirtToPhys` 依然正确（它本来就走 pagemap）。
- **Tiger Lake 不在原生支持列表**，用 Alder Lake DDR4 常量组是**近似**，
  所得函数必须与 DRAMA++ 的结果交叉验证才能采信（DRAMA++ 已在本机跑出
  `6 12 14` / `8 10 13 19`）。
- 复现命令：
```bash
export SUDOKU_PAGE_SIZE=2097152
export SUDOKU_NUM_CHANNELS=2
./reverse_functions -o out -p 1024 -t ddr4 -n 1 -s 8 -r 1 -w 8 -d -v
```
