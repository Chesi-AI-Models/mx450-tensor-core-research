#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 BIOS dump 的固件卷结构"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/vol1/ganlin/bios/bios_readonly.bin"
data = open(path, "rb").read()

print("大小: %d (%.1f MB)" % (len(data), len(data)/1048576))
print()
print("=== BIOS region 0x800000 头部 64 字节 ===")
chunk = data[0x800000:0x800040]
print(" ".join("%02X" % b for b in chunk))
print("ASCII:", "".join(chr(b) if 32 <= b < 127 else "." for b in chunk))

print()
print("=== Firmware Volume (_FVH) 统计 ===")
hits = []
p = 0
while True:
    p = data.find(b"_FVH", p)
    if p < 0:
        break
    hits.append(p - 40)
    p += 1
print("找到 %d 个 Firmware Volume" % len(hits))
for h in hits[:15]:
    if h + 40 <= len(data):
        size = int.from_bytes(data[h+32:h+40], "little")
        print("  FV @ 0x%08X  size=%d KB" % (h, size // 1024))

print()
print("=== 压缩相关线索 ===")
for n in (b"LZMA", b"Tiano", b"EFI_COMPRESS", b"LZNT", b"NVAR"):
    print("  %-14s %d" % (n.decode(), data.count(n)))

print()
print("=== 0x800000 之后前 8 个非零区域概览 ===")
for off in range(0x800000, min(len(data), 0x800000 + 0x200000), 0x10000):
    seg = data[off:off+16]
    if any(seg):
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in seg)
        print("  0x%08X  %s  |%s|" % (off, " ".join("%02X" % b for b in seg[:8]), asc[:8]))
