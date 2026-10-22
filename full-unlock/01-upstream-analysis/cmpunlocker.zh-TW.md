# CMPUnlocker 完整技術拆解

> 上游：`d3dx9/cmpunlocker`（GPL v2）。**原始碼不隨本倉庫分發**，用 `refs/refs-fetch-manifest.sh` 取得。
> 目標：NVIDIA CMP 170HX（GA100），8 GB → 64 GB，恢復 A100 算力
> 本文目的：搞清它的每一步，判斷哪些能用於 MX450

---

## 一、它解決的問題

**CMP 170HX 是一塊物理完整的 GA100（與 A100 同源矽晶片），被 OTP fuse + 韌體鎖成礦卡。**

| 項 | 出廠 | 解鎖後 |
|---|---|---|
| 顯示記憶體 | 8 GB（10 GB 版為 40 GB） | 64 GB HBM2e（5 × 16 GB） |
| SM 算力 | 被 PLM 限制 | 接近 A100 |
| PCIe | Gen1 ×4 | Gen2（best-effort） |

**關鍵前提：HBM2e 顆粒物理上就是 16 GB × 5 疊，只是被 strap 限制成每疊 2 GB。**

---

## 二、七步流程

```
1. Falcon BootROM 在驗證簽章之前，把 .fwsignature_ga100 節內容載入到 DMEM 0x800
   ↑ 這就是漏洞本身
2. 把該節內容替換成 24 DWORD 的 ROP 鏈
3. ROP 鏈執行：BAR0[write_addr] = write_value
4. 重複 4 次，打開 4 個 PLM 暫存器
5. PLM 打開後，host 驅動寫 CFG1/LMR（記憶體）+ SS0/SS1（算力）
6. 恢復原始 GSP 簽章，驅動檢測不到竄改
7. 驅動正常初始化，滿記憶體 + 滿 SM 時脈
```

**解鎖是 volatile 的（掉電失效），由 systemd daemon 每秒重放。**

---

## 三、ROP 鏈的結構（`tools/exploit_simulator.py`）

24 個 DWORD，其中兩個槽在執行時填充：

```python
ROP_GADGETS = [
    ("header",      0x1100, 0x00000007),   # 簽章頭
    ("canary_1",    0x5b40, 0xc0deca7e),
    ("write_value", 0xf754, None),         # ← 執行時填
    ("canary_2",    0xf758, 0xc0deca7e),
    ("gadget_1",    0xf75c, 0x00000cbd),   # Falcon 指令
    ("write_addr",  0xf76c, None),         # ← 執行時填
    ...
]
```

**核心：整條鏈的效果就是一次 `BAR0[write_addr] = write_value`。**

gadget 值是 Falcon 指令編碼（從韌體 DMEM 裡的既有程式碼找的），
所以 **gadget 位址與韌體版本強綁定**，不能跨版本／跨架構直接移植。

---

## 四、PLM 目標暫存器

```python
PLM_TABLE = [
    {"addr": 0x001FA7CC, "value": 0xFFFFf0ff, "name": "WPR_CFG"},
    {"addr": 0x009A0148, "value": 0xFFFFFFFF, "name": "FBPA"},
    {"addr": 0x001FA7C4, "value": 0xFFFFFFFF, "name": "WPR"},
    {"addr": 0x00823804, "value": 0xFFFFFFFF, "name": "FEAT"},
]
POST_PLM_WRITES = [
    {"addr": 0x0082381C, "value": 0x88888888, "name": "SS0 (SM speed)"},
    {"addr": 0x00823820, "value": 0x00000008, "name": "SS1"},
    {"addr": 0x009A0204, "value": 0x02669000, "name": "CFG1 (記憶體幾何)"},
    {"addr": 0x00100CE0, "value": 0x0000028A, "name": "LMR (memory rank)"},
]
```

**`0x0082381C`（SS0）就是 SM 限速暫存器 —— 本專案在階段 2 曾把這 8 個位址
全部在 Turing 的 GSP 韌體裡做小端搜尋，結果：8 個全部零命中。**

---

## 五、能否用於 MX450：逐條判定

| 環節 | 可用於 TU117？ | 原因 |
|---|---|---|
| `.fwsignature` 節存在 | ✅ | 本機 `gsp_tu10x.bin` 有 `.fwsignature_tu11x`（4096 位元組） |
| BootROM 先載入後驗證 | ❓ 未知 | 需要實測；Turing 的 BootROM 可能不同 |
| ROP gadget 位址 | ❌ | gadget 是 Falcon 指令編碼，與韌體版本強綁定 |
| **PLM 暫存器** | ❌ | **Turing 沒有 PLM 機制**（8 個位址零命中） |
| **CFG1/LMR 記憶體解鎖** | ❌ | **MX450 物理只有 2 GB，沒有可解鎖容量** |
| **SS0/SS1 算力解鎖** | ❌ | **TU117 沒有 TC，解鎖也無對象** |

**結論：CMPUnlocker 的方法論（ROP 注入繞過簽章）在 Turing 上「可能」可行，
但它要解鎖的目標（PLM、顯示記憶體容量、SM 算力）在 MX450 上全部不存在。**

**唯一有價值的潛在用途：ROP 鏈在 Falcon 上以 HS 模式執行，能存取 host 不可見的
`0x20000` 段（GR 內部暫存器）。這是突破 `HOST_IO_INDEX` 6 位限制的唯一技術路徑。**

但需要：① 實測 Turing BootROM 是否有同樣 bug；② 在 Turing DMEM 裡重新找 gadget；
③ 建構新鏈。工作量以週計，且屬於高風險操作（改韌體 + 注入程式碼）。

---

## 六、從中學到的三件事

1. **「破解」的前提是「有東西被封印」。** CMP 170HX 有 80 GB 物理顯示記憶體，MX450 只有 2 GB。
2. **簽章驗證的繞過點在「時序」而非「密碼學」。** BootROM 先載入後驗證，這個順序就是漏洞。
3. **ROP gadget 不可移植。** 它是從特定韌體的 DMEM 裡挖出來的指令片段。

---

## 七、授權

上游 `cmpunlocker` 為 **GPL v2**。本倉庫**不再內含其原始碼**，僅保留本檔案（原創分析，MIT）。
本專案其他部分為原創，不因引用該目錄而受 GPL 約束。
