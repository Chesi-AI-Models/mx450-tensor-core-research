# CMPUnlocker 完整技术拆解

> 上游：`d3dx9/cmpunlocker`（GPL v2）。**源码不随本仓库分发**，用 `refs/refs-fetch-manifest.sh` 获取。
> 目标：NVIDIA CMP 170HX（GA100），8 GB → 64 GB，恢复 A100 算力
> 本文目的：搞清它的每一步，判断哪些能用于 MX450

---

## 一、它解决的问题

**CMP 170HX 是一块物理完整的 GA100（与 A100 同源硅片），被 OTP fuse + 固件锁成矿卡。**

| 项 | 出厂 | 解锁后 |
|---|---|---|
| 显存 | 8 GB（10 GB 版为 40 GB） | 64 GB HBM2e（5 × 16 GB） |
| SM 算力 | 被 PLM 限制 | 接近 A100 |
| PCIe | Gen1 ×4 | Gen2（best-effort） |

**关键前提：HBM2e 颗粒物理上就是 16 GB × 5 栈，只是被 strap 限制成每栈 2 GB。**

---

## 二、七步流程

```
1. Falcon BootROM 在验证签名之前，把 .fwsignature_ga100 节内容加载到 DMEM 0x800
   ↑ 这就是漏洞本身
2. 把该节内容替换成 24 DWORD 的 ROP 链
3. ROP 链执行：BAR0[write_addr] = write_value
4. 重复 4 次，打开 4 个 PLM 寄存器
5. PLM 打开后，host 驱动写 CFG1/LMR（内存）+ SS0/SS1（算力）
6. 恢复原始 GSP 签名，驱动检测不到篡改
7. 驱动正常初始化，满内存 + 满 SM 时钟
```

**解锁是 volatile 的（掉电失效），由 systemd daemon 每秒重放。**

---

## 三、ROP 链的结构（`tools/exploit_simulator.py`）

24 个 DWORD，其中两个槽在运行时填充：

```python
ROP_GADGETS = [
    ("header",      0x1100, 0x00000007),   # 签名头
    ("canary_1",    0x5b40, 0xc0deca7e),
    ("write_value", 0xf754, None),         # ← 运行时填
    ("canary_2",    0xf758, 0xc0deca7e),
    ("gadget_1",    0xf75c, 0x00000cbd),   # Falcon 指令
    ("write_addr",  0xf76c, None),         # ← 运行时填
    ...
]
```

**核心：整条链的效果就是一次 `BAR0[write_addr] = write_value`。**

gadget 值是 Falcon 指令编码（从固件 DMEM 里的已有代码找的），
所以 **gadget 地址与固件版本强绑定**，不能跨版本/跨架构直接移植。

---

## 四、PLM 目标寄存器

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
    {"addr": 0x009A0204, "value": 0x02669000, "name": "CFG1 (内存几何)"},
    {"addr": 0x00100CE0, "value": 0x0000028A, "name": "LMR (memory rank)"},
]
```

**`0x0082381C`（SS0）就是 SM 限速寄存器 —— 本项目在阶段 2 曾把这 8 个地址
全部在 Turing 的 GSP 固件里做小端搜索，结果：8 个全部零命中。**

---

## 五、能否用于 MX450：逐条判定

| 环节 | 可用于 TU117？ | 原因 |
|---|---|---|
| `.fwsignature` 节存在 | ✅ | 本机 `gsp_tu10x.bin` 有 `.fwsignature_tu11x`（4096 字节） |
| BootROM 先加载后验证 | ❓ 未知 | 需要实测；Turing 的 BootROM 可能不同 |
| ROP gadget 地址 | ❌ | gadget 是 Falcon 指令编码，与固件版本强绑定 |
| **PLM 寄存器** | ❌ | **Turing 没有 PLM 机制**（8 个地址零命中） |
| **CFG1/LMR 内存解锁** | ❌ | **MX450 物理只有 2 GB，没有可解锁容量** |
| **SS0/SS1 算力解锁** | ❌ | **TU117 没有 TC，解锁也无对象** |

**结论：CMPUnlocker 的方法论（ROP 注入绕过签名）在 Turing 上"可能"可行，
但它要解锁的目标（PLM、显存容量、SM 算力）在 MX450 上全部不存在。**

**唯一有价值的潜在用途：ROP 链在 Falcon 上以 HS 模式运行，能访问 host 不可见的
`0x20000` 段（GR 内部寄存器）。这是突破 `HOST_IO_INDEX` 6 位限制的唯一技术路径。**

但需要：① 实测 Turing BootROM 是否有同样 bug；② 在 Turing DMEM 里重新找 gadget；
③ 构造新链。工作量以周计，且属于高风险操作（改固件 + 注入代码）。

---

## 六、从中学到的三件事

1. **"破解"的前提是"有东西被封印"。** CMP 170HX 有 80 GB 物理显存，MX450 只有 2 GB。
2. **签名验证的绕过点在"时序"而非"密码学"。** BootROM 先加载后验证，这个顺序就是漏洞。
3. **ROP gadget 不可移植。** 它是从特定固件的 DMEM 里挖出来的指令片段。

---

## 七、许可证

上游 `cmpunlocker` 为 **GPL v2**。本仓库**不再内含其源码**，仅保留本文件（原创分析，MIT）。
本项目其他部分为原创，不因引用该目录而受 GPL 约束。
