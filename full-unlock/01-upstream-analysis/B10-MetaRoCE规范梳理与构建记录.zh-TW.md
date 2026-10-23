# B10 MetaRoCE（含 MRC 協定）—— 規範梳理 + 建置記錄

## 一、專案資訊

| 項 | 值 |
|---|---|
| 倉庫 | `github.com/facebookincubator/metoce` → 實際為 `facebookincubator/meta**roce**` |
| 授權 | Apache-2.0 |
| 語言 | C++ |
| 規模 | 10 MB（原始碼）/ 規範 **6095 行** `spec/MetaRoCE-Spec-1.0.md` + PDF |
| 元件 | `spec/`（規範）、`libsoftmetaroce/`（軟體傳輸 + verbs provider）、`rdma_gen/`（RDMA 流量產生器 + Thrift 控制面）、`build/`（Meta getdeps 建置系統） |
| 定位 | 面向**百萬 GPU 叢集**、執行在**會丟包的乙太網路**上的 RDMA 傳輸協定，取代標準 RoCEv2 |
| 生態關係 | 與本專案 B9 的 Open-RDMA 互補：Open-RDMA 提供開源 RoCEv2 全端，MetaRoCE 提供**多路徑 + 抗丟包**的新傳輸層 |

## 二、四個核心機制（規範 §1.3）

| 機制 | 內容 | 為什麼重要 |
|---|---|---|
| **Tag 連線復用** | 一個 QP 承載多條獨立訊息流（Tag），各有自己的序號空間與完成順序 | 一條流因丟包／接收端未就緒卡住，**不阻塞**其他流 |
| **顯式多路徑** | 發送端在一個連線內維護多條 flow，每條有獨立滑動視窗 + SACK 狀態；靠改封包頭 entropy 走不同 ECMP 路由 | flow 狀態用**定長位圖**，記憶體有界；**能區分「這條 flow 掛了」和「暫時壅塞」**（盲目噴灑做不到）；頻寬延遲積變大時**加 flow 而不是加大單流視窗** |
| **鬆弛順序交付** | 資料包亂序到達即直接寫入記憶體最終位置；完成通知按序給應用 | 每個包自帶訊息內偏移；Write 還帶位址／長度／rkey；**不需要網卡上隨 BDP 增長的重排序緩衝** |
| **接收端驅動限速的壅塞控制** | 基線 AIMD 回應 ECN 與丟包 + 接收端在 ACK 裡回傳「分給你的入站頻寬份額」 | incast 場景**1–2 個 RTT 收斂**，徹底不依賴 PFC |

## 三、支援的操作（規範 §1.5）

1. SEND / RECV（帶／不帶 immediate、帶／不帶 Tag）
2. RDMA Write（帶／不帶 immediate）
3. RDMA Read（帶／不帶 immediate）
4. Atomics：compare-and-swap、fetch-and-add（opcode 為位元運算等預留）

## 四、封包頭格式（規範 §8，可直接用於從零實作）

**Base Header**（4 位元組對齊，MSB 序）：

```
 0         1         2         3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+---------------+---------------+-------+-----------------------+
|    Opcode     |   Reserved    |Version|        FlowID         |
+---------------------------------------------------------------+
|                        Destination QP                         |
+---------------------------------------------------------------+
|                           Source QP                           |
+---------------------------------------------------------------+
```

- 取 FlowID/Opcode 的官方範例程式碼：
  ```c
  uint32_t w = ntohl(((uint32_t *)p)[0]);
  int path_id = w & 0xfff;      // 低 12 位 = FlowID
  int opcode  = w >> 24;        // 高 8 位 = Opcode
  ```
- Opcode：RoCEv2 opcode 的 bits 7:5 用 `110` 表示 Meta-RoCE Connected Mode（廠商自訂區），
  bits 4:0 沿用 RoCEv2 模式；RoCEv2 的 first/middle/last 三態中**只採用 first**，其餘保留。
- Meta-RoCE **自訂** METH（資料）、SAETH（選擇性 ACK）、RNR-Cancel 頭；
  **借用 RoCEv2** 的 RETH（讀寫）、ImmDt、IETH、AtomicETH、AtomicAckETH。
- 所有包（資料／ACK／控制）尾部都有 **32 位 mCRC**，覆蓋頭 + 載荷 + padding；
  mCRC 是**位元組粒度**，因此接收端允許載荷不按 4 位元組對齊。
- 發送端需把載荷 pad 到 4 位元組倍數（直跑乙太網路時還要滿足最小訊框長），pad 位元組數記在資料頭的 Padding 欄位。
- 保留欄位：發送方必須寫 0，**接收方不得因此拒包**。

## 五、公開效能資料（Meta 口徑，64 節點 AMD GPU 叢集，RCCL 集合通訊）

| 指標 | RoCEv2 | MetaRoCE |
|---|---|---|
| 0% 丟包吞吐 | 基線 | 更高 |
| **1% 丟包** | 明顯退化 | **≈86% 吞吐** |
| **10% 丟包** | 崩 | **仍可用** |
| 多平面擴展 | 有限 | **線性（4/8 平面已驗證，最多 4000 條並行連線）** |
| 單平面失效恢復 | 人工／維運 | **自主重分配，應用無感** |

它只要求網路提供兩件現網交換機本來就有的能力：**ECN 標記 + ECMP**；
不要求包裁剪、帶內遙測、信用流控、交換機側噴灑 → **拓撲無關**（胖樹／多平面／深緩衝／淺緩衝／不可控的廠商雲均可）。

## 六、建置嘗試（getdeps）

上游唯一建置入口：

```bash
python3 build/fbcode_builder/getdeps.py build metaroce --src-dir .
# 即倉庫根目錄的 ./build.sh
```

**硬限制**：預設編譯器必須是 **gcc 而非 clang**，否則 **folly/coro 不會被建置**。
依賴鏈：`folly`（含 coro）、`glog`、`gtest`、`thrift`（rdma_gen 控制面）+ gflags 等。

本機條件：

| 項 | 值 |
|---|---|
| gcc | 12.2.0（Debian 12）✓ 滿足「必須是 gcc」 |
| cmake | 3.25.1 |
| 磁碟 | /root 空間充足（原始碼僅 10MB，但 getdeps 會建置 folly 等，需數 GB） |
| 網路 | NAS 有外網（見 B9 的糾正）✓ |

建置日誌：`/root/b10-build.log`（NAS）

### 6.1 第一次建置：卡在 libaio 下載 404（已定位根因並修復）

23m44s 後失敗，gflags / glog 均**已成功編譯**，卡在：

```
Failed to download https://releases.pagure.org/libaio/libaio-0.3.113.tar.gz
HTTP Error 404: Not Found
```

**聯網查到的根因**（不是網路問題）：
> **pagure.io 程式碼託管在 2026 年中關停**，libaio 的歸檔下載路徑隨之失效。
> libaio 上游遷到 Codeberg：`jmoyer/libaio`（Jeff Moyer 維護），
> 發布資產 `https://codeberg.org/jmoyer/libaio/releases/download/libaio-0.3.113/libaio-0.3.113.tar.gz`。
> 注意 Codeberg 的 tarball 與舊 pagure 歸檔**不是同一個檔案**（md5 `7d5be185…` vs 舊 `605237f3…`），
> 因為 pagure 是 git-archive 打包，目錄佈局不同。

而本倉庫 `build/fbcode_builder/manifests/libaio` 裡釘死的是舊包的雜湊：
```toml
[manifest]
name = libaio
[debs]
libaio-dev          # ← 關鍵：清單自己宣告了可用的系統套件
[download]
url = https://releases.pagure.org/libaio/libaio-0.3.113.tar.gz
sha256 = 2c44d1c5fd0d43752287c9ae1eb9c023f04ef848ea8d4aafa46e9aedb678200b
```

**修復方案（用上游自己支援的機制，不改任何 pin 值）**：
```bash
apt install -y libaio-dev
python3 build/fbcode_builder/getdeps.py --allow-system-packages build metaroce --src-dir .
```
即讓 getdeps 走 `[debs]` 分支用系統 `libaio-dev`，完全跳過這個壞掉的下載。

> 教訓（與 2026-10-20 01:46 那條一致）：**報錯第一步是搜尋錯誤訊息**。
> 若不搜，會誤以為是「NAS 網路問題」或去手動改雜湊 —— 而真正原因是**上游程式碼託管站關停**。

### 6.2 建置鏈路一共四個坑（全部已定位，含修法）

| # | 現象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `libaio-0.3.113.tar.gz` **404** | **pagure.io 2026 年中關停**，歸檔路徑失效 | `apt install libaio-dev` + `--allow-system-packages`（清單自帶 `[debs] libaio-dev`） |
| 2 | `automake-1.16.1.tar.gz` **403** | `ftpmirror.gnu.org` 對 getdeps 的 UA/IP 回傳 403 | 從 `cdimage.debian.org/mirror/gnu.org/gnu/...` 取同檔案，**按清單 sha256 校驗**後放進 getdeps `downloads/` 目錄（檔名格式 `<dep>-<tarball名>`） |
| 3 | 同類 403 會反覆出現 | 這是**系統性**問題，不是單包 | 一次性腳本：解析全部清單裡指向 `ftpmirror.gnu.org`／`ftp.gnu.org` 的 `url` + `sha256`，**批次補齊**（本次 5 個：binutils-2.43 / gperf-3.3 / libtool-2.4.6 / ncurses-6.3，全部雜湊匹配） |
| 4 | folly 連結失敗：`relocation R_X86_64_PC32 ... recompile with -fPIC`、`relocation against nil_eventop` | `libevent` 清單裡 **`EVENT__LIBRARY_TYPE = STATIC` 且無 PIC 定義** → `libevent.a` 非 PIC，不能鏈進 `libfolly.so` | ① 用 getdeps 自帶的 **`--shared-lib`**（"Build deps as static-**PIC** archives"）；② 仍不夠時，給 `manifests/libevent` 的 `[cmake.defines]` 顯式加 **`CMAKE_POSITION_INDEPENDENT_CODE = ON`**；③ **必須清掉 `build/` 與 `installed/` 快取**（getdeps 按 hash 目錄復用舊產物，不清會一直用非 PIC 的舊 `.a`；保留 `downloads/` 免得重下） |

**踩坑要點（寫死）**：
- 清單檔名是**無副檔名**的（`manifests/libevent` 不是 `libevent.toml`），`grep "*.toml"` 會全部漏掉。
- 清快取時**保留 `downloads/`**：裡面是校驗過雜湊的原始 tarball，刪了要重下一遍（binutils 就 28MB）。
- 每次改清單前先 `cp` 備份（本次備份到 `/root/libevent.manifest.bak`）。
- 判斷「是快取髒了還是真錯」：看報錯裡的 `installed/<dep>-<hash>/` 雜湊目錄**有沒有變**。
  雜湊不變 = 復用了舊產物。

## 七、與 MX450 專案的關係（誠實評估）

| 維度 | 結論 |
|---|---|
| 能否在本機跑通 | 取決於 getdeps 能否在 Debian 12 + gcc12 下把 folly/coro 建置出來 |
| **對 GPU 算力有沒有幫助** | **沒有**。它是網路傳輸層，不改變 MX450 的 SM／顯示記憶體／時鐘 |
| 對 F3「重寫網路底層」的意義 | **直接相關** —— F3 一直是 ❌ 未碰，MetaRoCE 的規範（序號層級、SACK 位圖、亂序交付、接收端限速）**正是「網路底層」的實作藍圖** |
| 與 B9 Open-RDMA 的組合 | Open-RDMA 給「開源 verbs 堆疊 + mock 裝置」，MetaRoCE 給「多路徑抗丟包傳輸語意」——兩者疊起來才是一條完整可改的軟 RDMA 路徑 |

**→ 因此 B10 的正確交付不是「編譯通過」一句話，而是：**
1. 真跑上游建置（進行中）；
2. **讀懂規範並用自研程式碼實作最小可跑的 MRC 語意**（Tag 多流 + 每 flow 獨立視窗 + 256 位 SACK 位圖 + 亂序落盤 + 接收端速率提示），
   直接作為 **F3「重寫網路底層」** 的第一塊磚。

## 八、相關搜尋要點（聯網核對）

- Meta 於 2026-08-24 公開設計細節，2026-10 OCP 高峰會放出**完整規範 + `libsoftmetaroce` 參考實作 + 生產合規測試集**。
- MRC = **Multipath Reliable Connection**，由 OpenAI / AMD / Broadcom / Microsoft / NVIDIA 共同提出；
  Open-RDMA 專案也聲明其多路徑傳輸借鑑 MRC 機制（見 B9 的 `open-rdma` 主倉 README）。
- 所以 B10 實際上要把**兩個專案對齊看**：MRC 是協定語意，MetaRoCE 是 Meta 的實作，Open-RDMA 是另一套開源實作。

## 九、fabric-lib（perplexityai/pplx-garden）—— 2026-10-20 真跑記錄

### 9.1 定位

fabric-lib **不是獨立倉庫**，開源於 `github.com/perplexityai/pplx-garden`
（MLSys 2026 oral，arXiv:2510.27656，MIT，C++/Rust）。
統一 ConnectX/EFA 兩家 RDMA 的 P2P 介面：WRITEIMM + IMM COUNTER 完成通知（無順序假設）、
多 NIC 聚合、透明多路徑。公開資料：ConnectX-7 與 EFA 上 400 Gbps。

### 9.2 建置鏈（本機 NAS 全部真跑通過）

| 步驟 | 結果 |
|---|---|
| `cargo build --release --bin fabric-debug` | 卡在 `cuda_profiler_api.h` 缺失（本機 CUDA 12.9 精簡安裝無此標頭） |
| 修 1：`cudart-sys/wrapper.h` 用 `#if __has_include` 包住該標頭 | 越過（備份 `/root/wrapper.h.bak.cudart-sys`） |
| 修 2：`GDRAPI_HOME` 缺失 → 從原始碼裝 **GDRCopy v2.5**（`make prefix=/usr/local CUDA=/usr/local/cuda install`） | 越過 |
| 修 3：`FI_MR_DMABUF`／`FI_OPT_CUDA_API_PERMITTED` 未解析 → 這些是 **AWS EFA 擴充常數**，Debian 標準 libfabric 沒有 → 裝 **AWS efa-installer**（`--skip-kmod --skip-limit-conf --no-verify`，只裝使用者態，libfabric 落到 `/opt/amazon/efa`） | **編譯成功：`target/release/fabric-debug` 2,970,264 B ELF** |

### 9.3 冒煙執行（誠實記錄）

```
$ fi_info -t FI_EP_RDM     # tcp provider 可見 enp45s0/wlp47s0/blue3；bluerdma0-3 仍在
$ ./fabric-debug 0 1
Error: FabricLibError: No visible NICs
```

**結論（與 B8 Sudoku 同類的「真實現實」）**：
- 介面／依賴／編譯鏈路**全部真跑通**（不是「判斷用不了」）；
- 執行時 fabric-lib 的 NIC 列舉認 EFA／標準 verbs 裝置，**不認 B9 的 mock `bluerdma`**，
  本機又沒有 ConnectX/EFA → 資料通路在本機走不到。
- 有價值的可執行餘量：① 讓 fabric-debug 走 B9 的 bluerdma 裝置（需在 fabric-lib 的
  NIC 列舉裡加 provider 識別 —— 屬於「讀原始碼用自己的程式碼重實作」的範疇）；② 純軟體
  部分（WRITEIMM/IMM COUNTER 語意）可用自研程式碼實作並驗證。

### 9.4 與 MX450 的關係（誠實評估）

fabric-lib 面向 **400G 資料中心 RDMA**，MX450 主機的 PCIe 3.0 x4（~4 GB/s）不在其目標面。
它的價值在本專案裡是：**F3 網路底層的第二塊磚 + 與 MRC/Open-RDMA 對齊的介面設計參照**，
不是算力提升路徑。
