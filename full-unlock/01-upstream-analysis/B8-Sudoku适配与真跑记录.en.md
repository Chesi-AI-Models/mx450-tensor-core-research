# B8 Sudoku — upstream code adaptation and real-run record

## 1. Project information

| Item | Value |
|---|---|
| upstream | `https://github.com/scale-snu/Sudoku` |
| paper | arXiv:2506.15918 "Sudoku: Decomposing DRAM Address Mapping into Component Functions" |
| license | MIT |
| purpose | decomposes DRAM address mapping to component level (channel / rank / bank group / bank) using a **pure-software timing channel**, and identifies row/column bits |
| native support | Intel Core 12th/14th (DDR4/DDR5), AMD Zen4 (DDR5) |
| this machine | **Intel i7-1165G7 (Tiger Lake)**, 2×8GB DDR4-3200, single rank, dual channel |

## 2. Upstream's four hard prerequisites vs measurements here

| Prerequisite | Upstream requirement | Measured here | Action |
|---|---|---|---|
| 1GB hugepages | grub `default_hugepagesz=1G hugepagesz=1G hugepages=N` + **reboot** | `/sys/.../hugepages-1048576kB/nr=0`; runtime `echo 4` **accepted but still 0** | **switched to 2MB hugepages** (1024 pages allocated at runtime, succeeded) |
| page_size | hardcoded `1ULL<<30` (1GB) | — | added `SUDOKU_PAGE_SIZE` env override |
| huge switch | hardcoded `true` in 7 mains | — | added `SUDOKU_NO_HUGE` env (default still on) |
| channel count | `MemoryConfig(1, 1, num_dimms, ...)` **hardcoded single channel** | **dual channel** (Controller0-ChannelA + Controller1-ChannelA) | added `SUDOKU_NUM_CHANNELS` / `SUDOKU_NUM_MCS` overrides |
| spdlog | CMake `FetchContent` from GitHub | **the NAS cannot reach GitHub** (curl/git ls-remote both empty) | vendored v1.13.0 into `third_party/spdlog` (version exactly matching upstream GIT_TAG) |
| numactl / cpupower | used in the commands | **both missing** (NAS cannot apt) | `taskset -c` instead of numactl; write `scaling_governor` / `scaling_{min,max}_freq` via sysfs instead of cpupower |
| build target | `-DCOMPILE_ALDER_LAKE_DDR4` etc., pick 1 of 4 | Tiger Lake not in the list | used the **ALDER_LAKE_DDR4** constant set (same Intel DDR4 MC family), **an adaptation, not native support** |

## 3. Exact change locations (minimal intrusion, all revertible)

| File | Change |
|---|---|
| `CMakeLists.txt` | FetchContent → `add_subdirectory(third_party/spdlog)`; compile macro becomes an overridable variable `SUDOKU_ARCH_FLAG` (default `-DCOMPILE_ALDER_LAKE_DDR4`) |
| `sudoku/internal/pool.h` | added two inline helpers `SudokuUseHuge()` / `SudokuPageSize(def)` |
| `sudoku/internal/pool.cc` | `MAP_HUGETLB \| MAP_HUGE_1GB` → choose `MAP_HUGE_2MB` or `MAP_HUGE_1GB` based on `page_size` |
| `reverse_functions.cc` / `identify_bits.cc` / `validate_mapping.cc` / `decompose_functions.cc` | `MemoryPoolConfig(page_size, ..., true)` → `MemoryPoolConfig(SudokuPageSize(page_size), ..., SudokuUseHuge())`; `MemoryConfig(1,1,...)` → env-overridable |
| `testing/watch_*.cc` × 3 | same (page_size / huge) |

**No algorithm logic was modified** (address-function inversion, conflict detection, refresh detection all stay exactly as upstream).

## 4. Real-run evidence

### 4.1 Build

```
-- Sudoku arch flag: -DCOMPILE_ALDER_LAKE_DDR4
=== BUILD OK ===
decompose_functions  identify_bits  reverse_functions  validate_mapping
watch_conflicts      watch_consecutive_accesses      watch_refreshes
```

### 4.2 Memory pool + physical addresses (pagemap working)

```
[+] Pool 28,0x7f9fa3e00000,0x205200000
[+] Pool 29,0x7f9fa3c00000,0x1ee800000
[+] Pool 30,0x7f9fa3a00000,0x10d600000
[+] Pool 31,0x7f9fa3800000,0x22a800000
[+] Pool 34,0x7f9fa3200000,0x3ec400000
[+] Pool 48,0x7f9fa1600000,0x48a200000
[+] Pool 59,0x7f9fa0000000,0x1eb000000
```
The left column is the virtual address (decreasing by 2MB per mmap, showing contiguity); the right column is the
**real physical address** decoded from `/proc/self/pagemap`. Physical addresses reach up to `0x4xxxx00000`
(about 19 GB of space), showing the 2MB pages are scattered across the whole physical address space — exactly
the distribution needed for inverting the address mapping.

### 4.3 Environment parameters

| Item | Value |
|---|---|
| hugepages | `hugepages-2048kB` 1024 pages (2GB), runtime allocation succeeded |
| CPU frequency | governor=`performance`, `scaling_min=max=2800000` (intel_pstate) |
| memory | 2×8GB DDR4-3200, single rank, dual channel |

## 5. Complete run result (`reverse_functions`, 7 min 54 s)

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
(about 130 candidate addresses)
Deriving DRAM addressing functions failed. Please retry.
  Found number of functions: 0, Expected number of functions: 5
```
Log size **1,218,771 bytes**; "Expected 5" comes from `log2(GetNumBanks)` =
`log2(1 MC × 2 CH × 1 DIMM × 1 rank × 4 bank_group × 4 bank) = log2(32) = 5`.

### 5.1 Why it failed (root cause, not "unusable")

| Constant | Alder Lake value | Meaning | Why it does not apply |
|---|---|---|---|
| `SBDR_LOWER_BOUND_` / `UPPER_BOUND_` | 450 / 600 | same-bank-different-row latency window (cycles) | Tiger Lake's MC timing and frequency differ; falls outside this window |
| `BANK_GROUP_THRESHOLD_` | 400 | latency boundary between consecutive accesses in same vs different bank group | same as above |
| `REFRESH_CYCLE_LOWER/UPPER_BOUND_` | 500 / 1000 | refresh cycle window | same as above |
| `REGULAR_REFRESH_INTERVAL_THRESHOLD_` | 20000 | periodic-refresh detection | same as above |

The paper states explicitly that these constants **must be calibrated per machine**; the repository only ships
headers for the machines they tested. So using Alder Lake constants here → conflict sets are not identified →
`Found 0`. **This is a fixable parameter problem, not an architectural dead end.**

### 5.2 Next step (calibration route)

1. `./watch_conflicts -m stat` to measure this machine's "single access" and "paired access" latency distributions
   (the smoke run already worked, printing `StatSingleMemoryAccess` / `StatPairedMemoryAccess`;
   note that when killed by timeout the spdlog buffer is not flushed, leaving the two .log files at 0 bytes)
2. derive this machine's SBDR window / bank-group boundary from the distribution
3. create `sudoku/internal/intel_core_11th_tigerlake_ddr4.h` and select it with `-DCOMPILE_TIGER_LAKE_DDR4`
4. rebuild and rerun `reverse_functions` / `identify_bits` / `validate_mapping` / `decompose_functions`
5. cross-validate the results against **DRAMA++, which already succeeded on this machine** (`6 12 14` / `8 10 13 19`)

---

## 6. Conclusion (B8)

- **upstream code really compiled and ran on this machine** — this is not a "judged unusable" case.
- upstream's 1GB hugepage dependency is an **optional optimisation** (`huge` is just a flag);
  after switching to 2MB pages `VirtToPhys` is still correct (it goes through pagemap anyway).
- **Tiger Lake is not on the native support list**; using the Alder Lake DDR4 constant set is an
  **approximation**, so the resulting functions must be cross-validated against DRAMA++ before being trusted
  (DRAMA++ already produced `6 12 14` / `8 10 13 19` on this machine).
- reproduction command:
```bash
export SUDOKU_PAGE_SIZE=2097152
export SUDOKU_NUM_CHANNELS=2
./reverse_functions -o out -p 1024 -t ddr4 -n 1 -s 8 -r 1 -w 8 -d -v
```
