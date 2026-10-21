# 三语翻译进度台账（i18n）

> 目标：把所有原创文档翻译为 **简体中文 / 繁體中文 / English** 三版。

> 规则：`<name>.md`（zh-CN 基准）→ `<name>.zh-TW.md` → `<name>.en.md`，与原文件同目录。

> 排除：`refs/`（NVIDIA 官方头文件原文，不译）、已移除的 `vendor/`。


## 总览

- 需翻译文档：**50 篇**，合计 **342 KB**
- 已完成：**2 篇**（根 `README.md` → `.zh-TW.md` / `.en.md`，已同步最新章节）
- 待完成：**49 篇**

## 优先级分批

### 第 1 批（核心叙事，先做）
- [ ] `full-unlock/05-firmware-exploit/README.md` (33 KB)
- [x] `full-unlock/README.md` (6 KB) — zh-TW + en 完成

### 第 2 批（findings 实测记录）
- [ ] `findings/2026-10-19-nvidia-ko符号级逆向.md` (11 KB)
- [ ] `findings/2026-10-19-限制机制不在驱动层.md` (10 KB)
- [ ] `findings/2026-10-19-iowr地址空间归属判定.md` (8 KB)
- [ ] `findings/2026-10-19-Falcon-IO地址空间完整模型.md` (8 KB)
- [ ] `findings/2026-10-19-GPCCS微码与通信机制.md` (8 KB)
- [ ] `findings/2026-10-19-最终结论-MX450没有TensorCore.md` (7 KB)
- [ ] `findings/2026-10-18-阶段2-GSP固件逆向.md` (7 KB)
- [ ] `findings/2026-10-18-阶段1-寄存器调查.md` (5 KB)
- [ ] `findings/2026-10-19-段归属判定失败与边界确认.md` (4 KB)
- [ ] `findings/2026-10-18-阶段1-补记5-GSP-vBIOS证书验证.md` (4 KB)
- [ ] `findings/2026-10-18-阶段3-P2-反汇编还原签名.md` (4 KB)
- [ ] `findings/2026-10-18-阶段4-FECS微码dump成功.md` (4 KB)
- [ ] `findings/2026-10-18-阶段3-P2-读通NVIDIA函数.md` (4 KB)
- [ ] `findings/2026-10-18-阶段3-PRAMIN读取成功.md` (4 KB)
- [ ] `findings/2026-10-18-阶段1-补记4-BIOS分析.md` (4 KB)
- [ ] `findings/2026-10-18-阶段3-PRAMIN读取受阻.md` (4 KB)
- [ ] `findings/2026-10-18-阶段1-补记2-ioctl通道.md` (4 KB)
- [ ] `findings/2026-10-18-阶段3-P2-矛盾解决.md` (4 KB)
- [ ] `findings/2026-10-18-阶段4-FECS微码反汇编.md` (3 KB)
- [ ] `findings/2026-10-18-阶段3-P2-尾调用确认.md` (3 KB)
- [ ] `findings/2026-10-18-阶段3-P1-kprobe.md` (3 KB)
- [ ] `findings/2026-10-18-阶段3-PRAMIN通道.md` (3 KB)
- [ ] `findings/2026-10-18-阶段3-FECS安全模式确认.md` (3 KB)
- [ ] `findings/2026-10-18-阶段3-微码不在显存.md` (3 KB)
- [ ] `findings/2026-10-18-阶段3-Falcon安全模式判定前置.md` (3 KB)
- [ ] `findings/2026-10-18-阶段2-补记-显存访问闭合.md` (3 KB)
- [ ] `findings/2026-10-18-阶段1-补记3-CUPTI不可用.md` (3 KB)
- [ ] `findings/2026-10-18-阶段4-寄存器段归属分析.md` (3 KB)
- [ ] `findings/2026-10-18-阶段1-补记-TC计数查询.md` (3 KB)
- [ ] `findings/2026-10-18-最终状态与明日待办.md` (3 KB)

### 第 3 批（full-unlock 研究报告）
- [ ] `full-unlock/05-firmware-exploit/任务B-FWSEC测绘报告.md` (15 KB)
- [ ] `full-unlock/01-upstream-analysis/B10-MetaRoCE规范梳理与构建记录.md` (12 KB)
- [ ] `full-unlock/01-upstream-analysis/知乎-GPU逆向方法论与TensorCore微架构-归档.md` (10 KB)
- [ ] `full-unlock/01-upstream-analysis/B13-shadPS4-GCN转译器真跑记录.md` (8 KB)
- [ ] `full-unlock/04-results/GEMM-路径破解实测报告.md` (7 KB)
- [ ] `full-unlock/04-results/T2-acceleration-round1.md` (7 KB)
- [ ] `full-unlock/01-upstream-analysis/cmpunlocker-实测记录.md` (6 KB)
- [ ] `full-unlock/04-results/D1-GEMM优化与D2锁频实测报告.md` (6 KB)
- [ ] `full-unlock/01-upstream-analysis/B8-Sudoku适配与真跑记录.md` (6 KB)
- [ ] `full-unlock/01-upstream-analysis/greenboost-adaptation.md` (5 KB)
- [ ] `full-unlock/03-toolkit/tilelang-实测记录.md` (5 KB)
- [ ] `full-unlock/03-toolkit/tinygrad-实测记录.md` (5 KB)
- [ ] `full-unlock/02-software-tensor-core/README.md` (4 KB)
- [ ] `full-unlock/01-upstream-analysis/cmpunlocker.md` (4 KB)

### 第 4 批（其余）
- [ ] `README.en.md` (14 KB)
- [ ] `README.zh-TW.md` (13 KB)
- [ ] `docs/03-内核模块路线图.md` (6 KB)
- [ ] `docs/02-技术路线图.md` (5 KB)

## 术语对照（保证三版一致）

| 简体 | 繁體（台湾） | English |
|---|---|---|
| 内核模块 | 核心模組 | kernel module |
| 显存 | 顯示記憶體 | VRAM |
| 固件 | 韌體 | firmware |
| 驱动 | 驅動程式 | driver |
| 微码 | 微碼 | microcode |
| 反汇编 | 反組譯 | disassembly |
| 寄存器 | 暫存器 | register |
| 实测 | 實測 | measured |
| 数据 | 資料 | data |
| 默认 | 預設 | default |
| 缓存 | 快取 | cache |
| 线程 | 執行緒 | thread |
