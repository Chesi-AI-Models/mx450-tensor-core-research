#!/usr/bin/env python3
import struct

F = "/vol1/ganlin/vram/vram_busy.bin"
d = open(F, "rb").read()
BASE = 0x7E000000
print("file size: %d (0x%X), VRAM base 0x%X" % (len(d), len(d), BASE))

sigs = {
    "FECS  完整头 (d0 00 20 00)": bytes.fromhex("d0002000" "00fe0400" "bda47e"),
    "GPCCS 完整头 (d0 00 14 00)": bytes.fromhex("d0001400" "00fe0400" "bda47e"),
    "公共 7 字节签名":            bytes.fromhex("00fe0400bda47e"),
    "Kepler 参考 (d0 00 10 00)":  bytes.fromhex("d0001000" "00fe0400" "bda47e"),
}

for name, pat in sigs.items():
    hits = []
    p = 0
    while True:
        p = d.find(pat, p)
        if p < 0:
            break
        hits.append(p)
        p += 1
        if len(hits) >= 10:
            break
    print("\n%s: %d 处" % (name, len(hits)))
    for h in hits[:8]:
        print("  VRAM 0x%08X (file 0x%X)" % (BASE + h, h))

# 如果找到，把整段 32KB 抠出来对比
print("\n=== 若命中，抠出候选微码并计算 MD5 ===")
import hashlib
pat = bytes.fromhex("00fe0400bda47e")
p = d.find(pat)
if p >= 0:
    start = p - 4
    for size in (32768, 16384):
        if start + size <= len(d):
            blob = d[start:start+size]
            print("  起点 0x%X, 取 %d 字节 -> MD5 %s" % (start, size, hashlib.md5(blob).hexdigest()))
    print("  参考: FECS  MD5 ea8eb3c46165b00421896eef45d1070a")
    print("  参考: GPCCS MD5 784c08e5a8fbfe9c8b300aa2b88e8116")
else:
    print("  未找到签名")

# 统计非零率
nz = sum(1 for b in d if b)
print("\n整片非零率: %.2f%%" % (100.0 * nz / len(d)))
