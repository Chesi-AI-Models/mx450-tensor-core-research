#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 FECS 微码反汇编中提取 iowr/iord 访问的寄存器地址"""
import re
from collections import Counter

lines = open("/vol1/ganlin/vram/fecs_imem.asm", encoding="utf-8", errors="replace").read().splitlines()

cur = {}
writes = []
reads = []

for ln in lines:
    m = re.search(r"mov \$r(\d+) 0x([0-9a-f]+)", ln)
    if m:
        cur[m.group(1)] = int(m.group(2), 16)
    m2 = re.search(r"iowr I\[\$r(\d+)\] \$r(\d+)", ln)
    if m2:
        writes.append(cur.get(m2.group(1)))
    m3 = re.search(r"iord \$r(\d+) I\[\$r(\d+)\]", ln)
    if m3:
        reads.append(cur.get(m3.group(2)))

print("iowr 总数: %d  (可解析地址 %d)" % (len(writes), sum(1 for x in writes if x)))
print("iord 总数: %d  (可解析地址 %d)" % (len(reads), sum(1 for x in reads if x)))
print()

cw = Counter(a for a in writes if a)
cr = Counter(a for a in reads if a)

print("=== iowr 访问最多的寄存器地址 ===")
for a, n in cw.most_common(30):
    print("  0x%08x  x%d" % (a, n))

print()
print("=== iord 访问最多的寄存器地址 ===")
for a, n in cr.most_common(15):
    print("  0x%08x  x%d" % (a, n))

print()
print("=== 按 64KB 段归类（看落在哪个硬件模块） ===")
seg = Counter((a >> 16) << 16 for a in writes if a)
for s, n in sorted(seg.items()):
    print("  0x%08x - 0x%08x : %d 次" % (s, s | 0xFFFF, n))
