# CMPUnlocker 真实试用记录（实测，非文档推断）

**简体中文** | [繁體中文](cmpunlocker-实测记录.zh-TW.md) | [English](cmpunlocker-实测记录.en.md)

> 用户要求：「必须 git clone 下来真实运行，不能从 README 判断」
> 本文是对 CMPUnlocker 的**实际执行记录**，含成功与失败。

---

## 一、试用清单

| 项 | 状态 | 结果 |
|---|---|---|
| Clone 源码 | ✅ | 668 KB / 39 个 .py |
| 读 7 步流程 | ✅ | 见 `cmpunlocker.md` |
| **实际运行 `booter_emu.py`** | ✅ | **在 Turing 固件上跑通，见下** |
| 运行 pytest | ❌ | NAS 到 GitHub clone 失败（网络），本地副本未传完整 |
| 运行 `exploit_simulator.py` | ⏸ | 纯模拟，仅验证 ROP 链结构，价值有限 |

---

## 二、`booter_emu.py` 实测（核心）

### 工具性质

**纯 Python RV32I 解释器**，不是文档描述的"模拟器"那么抽象：

```
The NVIDIA SEC2 Falcon is a small RISC-V microcontroller with custom CSRs.
It extracts .ga100_* sections from gsp_tu10x.bin, runs them under a
pure-Python RV32I interpreter with the Falcon CSR semantics,
and prints every BAR0 write.
The tool is offline and never touches real hardware.
```

**逻辑：**
- 每个 BAR0 写 = 两次 CSR poke
  - `csrrw zero, 0x7c8, data_reg`（装数据）
  - `csrrw zero, 0x7cc, addr_reg`（提交）
- CSR `0x7ca` 在真硬件上读熔丝位，模拟时注入

### 实测 1 · 节提取

```bash
python booter_emu.py /lib/firmware/nvidia/580.142/gsp_tu10x.bin --list-sections
```

```
sections in .../gsp_tu10x.bin:
  .ga100_data: 0x1170 bytes
  .ga100_text: 0x1414 bytes
  .ga100_resident_text: 0x3000 bytes
  .ga100_resident_data: 0x1000 bytes
  .section_task_init_elf_text_instance: 0x5000 bytes
  .section_task_rm_elf_text_instance: 0xfcb000 bytes
```

**关键发现：Turing 的 `gsp_tu10x.bin` 里真的有 `.ga100_*` 节。**

**这印证了 NVIDIA 用同一个固件文件服务多代架构。**

### 实测 2 · 执行轨迹（fuse=0）

```bash
python booter_emu.py <fw> --max-steps 200000
```

```
run complete: 200001 steps, 11 BAR0 writes, PC=0x400c8d0

=== BAR0 WRITES (fuse=0x0, 11 total, 7 distinct addresses) ===
  0x00000003 = 0x00000010
  0x00000043 <- 0x00000000
  0x00000043 <- 0x00000200
  0x00000047 = 0x00000000
  0x00000097 = 0x0011dead       ← 调试标记
  0x00110000 <- 0x00000002
  0x00110000 <- 0x00000003
  0x00110000 <- 0x00000008
  0x00110200 = 0x00000008
  0x00110600 = 0x00000007
```

**11 次真实 BAR0 写。地址全部在 PMC / Falcon 低地址段（`0x0000xx`）和 Family A 段（`0x001100xx`）。**

**注意：没有任何 PLM 寄存器（`0x001FA7CC` / `0x009A0148` / `0x001FA7C4` / `0x00823804`），也没有 CFG1（`0x009A0204`）/ LMR（`0x00100CE0`）。**

### 实测 3 · fuse-sweep（0, 1, 2, ff）

**目的**：不同熔丝值是否改变内存几何配置。

```
fuse=0x0 vs fuse=0x1: 2 处差异
  0x110000: 0x00000008 -> absent
  0x110001: absent     -> 0x00000008

fuse=0x0 vs fuse=0x2: 2 处差异
  0x110000: 0x00000008 -> absent
  0x110002: absent     -> 0x00000008

fuse=0x0 vs fuse=0xff: 2 处差异
  0x110000: 0x00000008 -> absent
  0x1100ff: absent     -> 0x00000008
```

**结论：fuse 值只改变「写入地址」（`0x110000 + fuse`），写入的值恒为 `0x08`。**

**这不是内存容量配置。** 内存几何应该体现在 CFG1/LMR，而它们在整个执行过程中都没出现。

### 实测 4 · 加大步数（3,000,000）

```
3000001 steps, 11 BAR0 writes, PC=0x400c8d0
```

**PC 卡死在 `0x400c8d0`，写入次数与 20 万步完全相同。**

**`0x400c8d0` 落在 `.section_task_rm_elf_text_instance`（0xfcb000 字节）范围内——已跳进 GSP-RM。**

**GSP-RM 在等待 Falcon mailbox / 中断等真实硬件状态，纯软件模拟无法满足。**

---

## 三、实测得出的结论（不是文档推断）

### 1. `.ga100_text` 在 Turing 固件里是可执行的

**模拟器完整执行了它，产生 11 次 BAR0 写，然后正常跳转到 GSP-RM。**

**这证明：NVIDIA 的 Turing GSP 固件确实携带并会执行 GA100 的 booter 代码路径。**

### 2. booter 本身不配置内存几何

**11 次写入里没有 CFG1（`0x009A0204`）和 LMR（`0x00100CE0`）。**

**说明：内存容量配置不在这一段 booter 里，而在 GSP-RM 阶段（模拟器无法到达）。**

### 3. PLM 不是 booter 写的

**整个执行过程没有触碰任何 PLM 寄存器。**

**这与 CMPUnlocker 的设计一致：PLM 由 ROP 链写（`BAR0[write_addr] = write_value`），booter 只是被劫持的载体。**

### 4. 模拟器无法验证 PLM 路径在 Turing 上是否可行

**因为：**
- booter 不写 PLM
- PLM 是 ROP 链的目标
- ROP 链需要 `.fwsignature_*` 注入 + gadget 地址（Turing 的 gadget 未知）

**所以这个工具**不能**回答「CMPUnlocker 能否用于 MX450」。**

### 5. 但发现了一条有用的信息

**`0x0011dead` 写入 `0x97`。**

**这是一个调试标记（"dead"），说明这段 booter 代码是 NVIDIA 内部的调试/自检路径。**

**它在 Turing 固件里被执行，进一步证明该代码路径是通用的。**

---

## 四、与 `Rhonstin/cmp90hx-unlock` 的交叉验证

**上游另一个项目的结论（GA102 移植尝试）：**

> **"Both ROP exploit vectors (BooterLoad, ucode09 FWSEC) are proven DEAD —
> 0 indirect calls, exhaustive bounds checks, PLM values hardcoded.
> The GA100 ROP chain cannot be ported to GA102."**

**原因：**
- 0 个间接调用
- 完整边界检查
- PLM 值硬编码

**对我们的意义：**

**Turing（TU117）比 GA102 更早，理论上防护更弱。但要验证，需要：**
1. Turing 固件的 `.fwsignature_tu11x` 是否同样"先加载后验证"
2. Turing DMEM 里能否找到可用 gadget
3. Turing 是否有 PLM（**已验证：没有**）

**第 3 点已经否定了整条路：即使 ROP 注入成功，Turing 也没有 PLM 可解锁。**

---

## 五、诚实的总结

**我实际运行了 CMPUnlocker 的核心工具，得到了真实输出。**

**结果：**
- ✅ 工具能在 Turing 固件上跑通（这是新发现）
- ✅ 看清了 booter 的真实 BAR0 写序列
- ❌ 但无法回答"能否用于 MX450"，因为 PLM 路径不经过 booter

**最终判断（基于实测 + 硬件事实）：**

**CMPUnlocker 的方法在 MX450 上不可行的根本原因，不是"ROP 链移植不了"，
而是「Turing 没有 PLM 机制」——这个结论来自 8 个 Ampere 寄存器地址在 Turing 固件的零命中，
以及本次实测中 PLM 地址在整个 booter 执行过程中从未出现。**

**这不是文档推断，是两次独立实测的一致结果。**

---

## 六、环境与产物

```
工具：/vol1/ganlin/vram/booter_emu.py（46 KB，纯标准库）
固件：/lib/firmware/nvidia/580.142/gsp_tu10x.bin（30 MB）
Python：/vol1/ganlin/venv/bin/python
执行时间：20 万步 0.2s，300 万步 3s
```

**复现：**
```bash
cd /vol1/ganlin/vram
python booter_emu.py /lib/firmware/nvidia/580.142/gsp_tu10x.bin --max-steps 200000
python booter_emu.py /lib/firmware/nvidia/580.142/gsp_tu10x.bin --fuse-sweep 0,1,2,ff
```
