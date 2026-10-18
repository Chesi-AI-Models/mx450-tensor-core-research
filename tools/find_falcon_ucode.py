#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在显存 dump 中搜索 Falcon 微码特征。

已知参考（gk20a Kepler）：
  fecs_inst.bin : d0 00 10 00 00 fe 04 00 bd a4 7e 2f 3a 00 f8 02
  gpccs_inst.bin: d0 00 08 00 00 fe 04 00 bd a4 7e d9 1e 00 f8 02
共同前缀：d0 00 ?? 00 00 fe 04 00 bd a4 7e  （12 字节，第 3 字节随栈顶变化）

策略：
  1. 搜 8 字节强特征 00 fe 04 00 bd a4 7e
  2. 搜 4 字节 d0 00 ?? 00（放宽）
  3. 对命中位置 dump 上下文，判断是否像 IMEM 代码段（指令密度）
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/vol1/ganlin/vram/vram_dump.bin"
data = open(path, "rb").read()
print("文件: %s  大小: %.1f MB" % (path, len(data) / 1048576))
print("=" * 66)

def find_all(pat, limit=20):
    hits, p = [], 0
    while True:
        p = data.find(pat, p)
        if p < 0:
            break
        hits.append(p)
        p += 1
        if len(hits) >= limit:
            break
    return hits

# 1) 8 字节强特征
sig8 = bytes.fromhex("00fe0400bda47e")
h8 = find_all(sig8, 30)
print("\n[1] 强特征 00fe0400bda47e : %d 处" % len(h8))
for h in h8[:15]:
    print("    0x%08X" % h)

# 2) 放宽：d0 00 ?? 00
print("\n[2] 放宽 d0 00 ?? 00 计数")
cnt = 0
locs = []
for i in range(len(data) - 4):
    if data[i] == 0xD0 and data[i+1] == 0x00 and data[i+3] == 0x00:
        cnt += 1
        if len(locs) < 15:
            locs.append(i)
print("    命中 %d 处，前 15:" % cnt)
for h in locs:
    print("    0x%08X  ctx: %s" % (h, data[h:h+16].hex(" ")))

# 3) 对首个强特征命中做上下文 dump
if h8:
    h = h8[0]
    start = max(0, h - 16)
    print("\n[3] 首个命中 0x%08X 上下文（%- 从 -16 开始）" % h)
    chunk = data[start:start+96]
    for i in range(0, len(chunk), 16):
        seg = chunk[i:i+16]
        print("    +0x%04X  %s  |%s|" % (
            start + i - start, " ".join("%02x" % b for b in seg),
            "".join(chr(b) if 32 <= b < 127 else "." for b in seg)))

# 4) 统计"看起来像代码"的 4KB 块（字节熵高、零少）
print("\n[4] 前 32MB 中指令密度较高的 4KB 块（非零率 > 80%）")
blocks = 0
for off in range(0, len(data), 0x1000):
    seg = data[off:off+0x1000]
    if len(seg) < 0x1000:
        break
    nz = sum(1 for b in seg if b)
    if nz > 0x1000 * 0.8:
        blocks += 1
        if blocks <= 12:
            print("    0x%08X  非零 %.1f%%" % (off, nz * 100.0 / 0x1000))
print("    合计 %d / %d 个 4KB 块" % (blocks, len(data) // 0x1000))
print("=" * 66)
