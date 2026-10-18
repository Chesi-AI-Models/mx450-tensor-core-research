#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 GSP 固件中出现的 GPU 寄存器空间地址常量。

思路：不再拿 Ampere 的确切地址去比对（那会漏掉 Turing 的真实地址），
而是统计所有落在 GPU MMIO 区间的 32 位值，看 Turing 用的是哪些地址段。

GPU 寄存器空间经验区间：
  0x00000000 - 0x02000000   (PMC/PFB/GR 等)
Ampere 已知锚点：
  SS0      0x0082381C
  FBPA_CFG 0x009A0204
  MMU_LMR  0x00100CE0
  WPR      0x001FA7C4
"""
import sys, struct
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/inner_gsp.elf"
data = open(path, "rb").read()
print("文件: %s  大小: %.1f MB" % (path, len(data)/1048576))
print("=" * 70)

# 4 字节对齐扫描
n = len(data) // 4
vals = struct.unpack_from("<%dI" % n, data, 0)

# 只保留 GPU 寄存器空间候选
cand = [v for v in vals if 0x1000 <= v < 0x02000000]

# 按高 16 位分桶（即 64KB 页），统计热点
pages = Counter(v >> 16 for v in cand)

print("\n[1] 命中最多的 16 个 64KB 页（地址段热点）")
print("    %-14s %-12s %s" % ("页 (v>>16)", "地址范围", "出现次数"))
for pg, cnt in pages.most_common(16):
    print("    0x%04X         0x%08X-0x%08X   %d"
          % (pg, pg << 16, (pg << 16) | 0xFFFF, cnt))

# 聚焦 Ampere 已知段，看 Turing 是否也在此
print("\n[2] Ampere 已知锚点所在页的命中情况")
anchors = {
    "SS0/SS1 段 0x0082xxxx": 0x0082,
    "FBPA 段   0x009Axxxx": 0x009A,
    "MMU 段    0x0010xxxx": 0x0010,
    "WPR 段    0x001Fxxxx": 0x001F,
}
for name, pg in anchors.items():
    print("  %-24s %d 次" % (name, pages.get(pg, 0)))

# 列出 0x0082 段的具体值（如果有）
print("\n[3] 0x0082xxxx 段的具体常量（若存在，说明 Turing 也有该寄存器带）")
seg82 = sorted({v for v in cand if (v >> 16) == 0x0082})
if seg82:
    for v in seg82[:20]:
        print("    0x%08X" % v)
else:
    print("    （无）")

print("\n[4] 全部候选常量总数: %d / %d 个 32 位字" % (len(cand), n))
print("=" * 70)
