# findings index — MX450 reverse-engineering measurement records (30 documents, all trilingual)

> This directory is an append-only measurement log. **Every document has three language versions:**
> - `<name>.md` — Simplified Chinese (baseline)
> - `<name>.zh-TW.md` — Traditional Chinese (Taiwan terminology)
> - `<name>.en.md` — English
>
> All register addresses, disassembly, MD5 digests and timing numbers are **identical across the three versions**
> (values are never translated). Document filenames stay in their original form so the links work.
> Translation progress tracker: [`../docs/i18n/TRANSLATION-STATUS.md`](../docs/i18n/TRANSLATION-STATUS.md)

---


## Stage 1 · user-space and firmware-layer reconnaissance

| Document | Size | Languages |
|---|---|---|
| [2026-10-18-阶段1-寄存器调查](2026-10-18-阶段1-寄存器调查.md) | 5 KB | [zh-CN](2026-10-18-阶段1-寄存器调查.md) · [zh-TW](2026-10-18-阶段1-寄存器调查.zh-TW.md) · [EN](2026-10-18-阶段1-寄存器调查.en.md) |
| [2026-10-18-阶段1-补记-TC计数查询](2026-10-18-阶段1-补记-TC计数查询.md) | 3 KB | [zh-CN](2026-10-18-阶段1-补记-TC计数查询.md) · [zh-TW](2026-10-18-阶段1-补记-TC计数查询.zh-TW.md) · [EN](2026-10-18-阶段1-补记-TC计数查询.en.md) |
| [2026-10-18-阶段1-补记2-ioctl通道](2026-10-18-阶段1-补记2-ioctl通道.md) | 4 KB | [zh-CN](2026-10-18-阶段1-补记2-ioctl通道.md) · [zh-TW](2026-10-18-阶段1-补记2-ioctl通道.zh-TW.md) · [EN](2026-10-18-阶段1-补记2-ioctl通道.en.md) |
| [2026-10-18-阶段1-补记3-CUPTI不可用](2026-10-18-阶段1-补记3-CUPTI不可用.md) | 3 KB | [zh-CN](2026-10-18-阶段1-补记3-CUPTI不可用.md) · [zh-TW](2026-10-18-阶段1-补记3-CUPTI不可用.zh-TW.md) · [EN](2026-10-18-阶段1-补记3-CUPTI不可用.en.md) |
| [2026-10-18-阶段1-补记4-BIOS分析](2026-10-18-阶段1-补记4-BIOS分析.md) | 4 KB | [zh-CN](2026-10-18-阶段1-补记4-BIOS分析.md) · [zh-TW](2026-10-18-阶段1-补记4-BIOS分析.zh-TW.md) · [EN](2026-10-18-阶段1-补记4-BIOS分析.en.md) |
| [2026-10-18-阶段1-补记5-GSP-vBIOS证书验证](2026-10-18-阶段1-补记5-GSP-vBIOS证书验证.md) | 4 KB | [zh-CN](2026-10-18-阶段1-补记5-GSP-vBIOS证书验证.md) · [zh-TW](2026-10-18-阶段1-补记5-GSP-vBIOS证书验证.zh-TW.md) · [EN](2026-10-18-阶段1-补记5-GSP-vBIOS证书验证.en.md) |

## Stage 2 · GSP firmware reverse engineering

| Document | Size | Languages |
|---|---|---|
| [2026-10-18-阶段2-GSP固件逆向](2026-10-18-阶段2-GSP固件逆向.md) | 7 KB | [zh-CN](2026-10-18-阶段2-GSP固件逆向.md) · [zh-TW](2026-10-18-阶段2-GSP固件逆向.zh-TW.md) · [EN](2026-10-18-阶段2-GSP固件逆向.en.md) |
| [2026-10-18-阶段2-补记-显存访问闭合](2026-10-18-阶段2-补记-显存访问闭合.md) | 3 KB | [zh-CN](2026-10-18-阶段2-补记-显存访问闭合.md) · [zh-TW](2026-10-18-阶段2-补记-显存访问闭合.zh-TW.md) · [EN](2026-10-18-阶段2-补记-显存访问闭合.en.md) |

## Stage 3 · opening channels (PRAMIN / kprobe / NVIDIA functions)

| Document | Size | Languages |
|---|---|---|
| [2026-10-18-阶段3-FECS安全模式确认](2026-10-18-阶段3-FECS安全模式确认.md) | 3 KB | [zh-CN](2026-10-18-阶段3-FECS安全模式确认.md) · [zh-TW](2026-10-18-阶段3-FECS安全模式确认.zh-TW.md) · [EN](2026-10-18-阶段3-FECS安全模式确认.en.md) |
| [2026-10-18-阶段3-Falcon安全模式判定前置](2026-10-18-阶段3-Falcon安全模式判定前置.md) | 3 KB | [zh-CN](2026-10-18-阶段3-Falcon安全模式判定前置.md) · [zh-TW](2026-10-18-阶段3-Falcon安全模式判定前置.zh-TW.md) · [EN](2026-10-18-阶段3-Falcon安全模式判定前置.en.md) |
| [2026-10-18-阶段3-P1-kprobe](2026-10-18-阶段3-P1-kprobe.md) | 4 KB | [zh-CN](2026-10-18-阶段3-P1-kprobe.md) · [zh-TW](2026-10-18-阶段3-P1-kprobe.zh-TW.md) · [EN](2026-10-18-阶段3-P1-kprobe.en.md) |
| [2026-10-18-阶段3-P2-反汇编还原签名](2026-10-18-阶段3-P2-反汇编还原签名.md) | 4 KB | [zh-CN](2026-10-18-阶段3-P2-反汇编还原签名.md) · [zh-TW](2026-10-18-阶段3-P2-反汇编还原签名.zh-TW.md) · [EN](2026-10-18-阶段3-P2-反汇编还原签名.en.md) |
| [2026-10-18-阶段3-P2-尾调用确认](2026-10-18-阶段3-P2-尾调用确认.md) | 4 KB | [zh-CN](2026-10-18-阶段3-P2-尾调用确认.md) · [zh-TW](2026-10-18-阶段3-P2-尾调用确认.zh-TW.md) · [EN](2026-10-18-阶段3-P2-尾调用确认.en.md) |
| [2026-10-18-阶段3-P2-矛盾解决](2026-10-18-阶段3-P2-矛盾解决.md) | 4 KB | [zh-CN](2026-10-18-阶段3-P2-矛盾解决.md) · [zh-TW](2026-10-18-阶段3-P2-矛盾解决.zh-TW.md) · [EN](2026-10-18-阶段3-P2-矛盾解决.en.md) |
| [2026-10-18-阶段3-P2-读通NVIDIA函数](2026-10-18-阶段3-P2-读通NVIDIA函数.md) | 4 KB | [zh-CN](2026-10-18-阶段3-P2-读通NVIDIA函数.md) · [zh-TW](2026-10-18-阶段3-P2-读通NVIDIA函数.zh-TW.md) · [EN](2026-10-18-阶段3-P2-读通NVIDIA函数.en.md) |
| [2026-10-18-阶段3-PRAMIN读取受阻](2026-10-18-阶段3-PRAMIN读取受阻.md) | 4 KB | [zh-CN](2026-10-18-阶段3-PRAMIN读取受阻.md) · [zh-TW](2026-10-18-阶段3-PRAMIN读取受阻.zh-TW.md) · [EN](2026-10-18-阶段3-PRAMIN读取受阻.en.md) |
| [2026-10-18-阶段3-PRAMIN读取成功](2026-10-18-阶段3-PRAMIN读取成功.md) | 4 KB | [zh-CN](2026-10-18-阶段3-PRAMIN读取成功.md) · [zh-TW](2026-10-18-阶段3-PRAMIN读取成功.zh-TW.md) · [EN](2026-10-18-阶段3-PRAMIN读取成功.en.md) |
| [2026-10-18-阶段3-PRAMIN通道](2026-10-18-阶段3-PRAMIN通道.md) | 3 KB | [zh-CN](2026-10-18-阶段3-PRAMIN通道.md) · [zh-TW](2026-10-18-阶段3-PRAMIN通道.zh-TW.md) · [EN](2026-10-18-阶段3-PRAMIN通道.en.md) |
| [2026-10-18-阶段3-微码不在显存](2026-10-18-阶段3-微码不在显存.md) | 3 KB | [zh-CN](2026-10-18-阶段3-微码不在显存.md) · [zh-TW](2026-10-18-阶段3-微码不在显存.zh-TW.md) · [EN](2026-10-18-阶段3-微码不在显存.en.md) |

## Stage 4 · microcode dump and disassembly

| Document | Size | Languages |
|---|---|---|
| [2026-10-18-阶段4-FECS微码dump成功](2026-10-18-阶段4-FECS微码dump成功.md) | 4 KB | [zh-CN](2026-10-18-阶段4-FECS微码dump成功.md) · [zh-TW](2026-10-18-阶段4-FECS微码dump成功.zh-TW.md) · [EN](2026-10-18-阶段4-FECS微码dump成功.en.md) |
| [2026-10-18-阶段4-FECS微码反汇编](2026-10-18-阶段4-FECS微码反汇编.md) | 4 KB | [zh-CN](2026-10-18-阶段4-FECS微码反汇编.md) · [zh-TW](2026-10-18-阶段4-FECS微码反汇编.zh-TW.md) · [EN](2026-10-18-阶段4-FECS微码反汇编.en.md) |
| [2026-10-18-阶段4-寄存器段归属分析](2026-10-18-阶段4-寄存器段归属分析.md) | 3 KB | [zh-CN](2026-10-18-阶段4-寄存器段归属分析.md) · [zh-TW](2026-10-18-阶段4-寄存器段归属分析.zh-TW.md) · [EN](2026-10-18-阶段4-寄存器段归属分析.en.md) |

## 2026-10-19 · architecture-level discoveries and convergence

| Document | Size | Languages |
|---|---|---|
| [2026-10-19-Falcon-IO地址空间完整模型](2026-10-19-Falcon-IO地址空间完整模型.md) | 8 KB | [zh-CN](2026-10-19-Falcon-IO地址空间完整模型.md) · [zh-TW](2026-10-19-Falcon-IO地址空间完整模型.zh-TW.md) · [EN](2026-10-19-Falcon-IO地址空间完整模型.en.md) |
| [2026-10-19-GPCCS微码与通信机制](2026-10-19-GPCCS微码与通信机制.md) | 8 KB | [zh-CN](2026-10-19-GPCCS微码与通信机制.md) · [zh-TW](2026-10-19-GPCCS微码与通信机制.zh-TW.md) · [EN](2026-10-19-GPCCS微码与通信机制.en.md) |
| [2026-10-19-iowr地址空间归属判定](2026-10-19-iowr地址空间归属判定.md) | 9 KB | [zh-CN](2026-10-19-iowr地址空间归属判定.md) · [zh-TW](2026-10-19-iowr地址空间归属判定.zh-TW.md) · [EN](2026-10-19-iowr地址空间归属判定.en.md) |
| [2026-10-19-nvidia-ko符号级逆向](2026-10-19-nvidia-ko符号级逆向.md) | 11 KB | [zh-CN](2026-10-19-nvidia-ko符号级逆向.md) · [zh-TW](2026-10-19-nvidia-ko符号级逆向.zh-TW.md) · [EN](2026-10-19-nvidia-ko符号级逆向.en.md) |
| [2026-10-19-最终结论-MX450没有TensorCore](2026-10-19-最终结论-MX450没有TensorCore.md) | 8 KB | [zh-CN](2026-10-19-最终结论-MX450没有TensorCore.md) · [zh-TW](2026-10-19-最终结论-MX450没有TensorCore.zh-TW.md) · [EN](2026-10-19-最终结论-MX450没有TensorCore.en.md) |
| [2026-10-19-段归属判定失败与边界确认](2026-10-19-段归属判定失败与边界确认.md) | 4 KB | [zh-CN](2026-10-19-段归属判定失败与边界确认.md) · [zh-TW](2026-10-19-段归属判定失败与边界确认.zh-TW.md) · [EN](2026-10-19-段归属判定失败与边界确认.en.md) |
| [2026-10-19-限制机制不在驱动层](2026-10-19-限制机制不在驱动层.md) | 10 KB | [zh-CN](2026-10-19-限制机制不在驱动层.md) · [zh-TW](2026-10-19-限制机制不在驱动层.zh-TW.md) · [EN](2026-10-19-限制机制不在驱动层.en.md) |

## Other

| Document | Size | Languages |
|---|---|---|
| [2026-10-18-最终状态与明日待办](2026-10-18-最终状态与明日待办.md) | 3 KB | [zh-CN](2026-10-18-最终状态与明日待办.md) · [zh-TW](2026-10-18-最终状态与明日待办.zh-TW.md) · [EN](2026-10-18-最终状态与明日待办.en.md) |

---

## Suggested reading order

1. **read the conclusion first**: [`2026-10-19-最终结论-MX450没有TensorCore`](2026-10-19-最终结论-MX450没有TensorCore.md)
2. **then the architecture**: [`2026-10-19-Falcon-IO地址空间完整模型`](2026-10-19-Falcon-IO地址空间完整模型.md)
3. **then which layer the restriction lives in**: [`2026-10-19-限制机制不在驱动层`](2026-10-19-限制机制不在驱动层.md)
4. **to see how it was done**: the stage 3 → stage 4 documents

## Hard facts quick reference

| Item | Value |
|---|---|
| FECS microcode | 32 KB, MD5 `ea8eb3c46165b00421896eef45d1070a` |
| GPCCS microcode | 16 KB, MD5 `784c08e5a8fbfe9c8b300aa2b88e8116` |
| FECS/GPCCS security mode | LIGHT SECURE (`security_model=2`) |
| Falcon ISA | fuc6, RV64 (`core_rev=6`) |
| Measured TC throughput | 0.60 TFLOPS (mma) vs 5.77 (FP16x2) → 9.6× slower |
| Conclusion | TU117 **has no Tensor Core**; HMMA is emulated by FP16 units |
