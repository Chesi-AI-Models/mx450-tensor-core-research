#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在系统 BIOS dump 中定位 NVIDIA vBIOS 镜像"""
import sys, struct

path = sys.argv[1] if len(sys.argv) > 1 else "/vol1/ganlin/bios/bios_readonly.bin"
data = open(path, "rb").read()
print("文件: %s" % path)
print("大小: %d 字节 (%.1f MB)" % (len(data), len(data)/1048576))
print("=" * 70)

# ---- 1. 找所有可能的 PCI Option ROM 起点（55 AA + 偏移 0x1a 处的 PCIR 指针） ----
print("\n[1] 扫描 PCI Option ROM 候选（55AA 开头）")
cands = []
pos = 0
while True:
    pos = data.find(b"\x55\xaa", pos)
    if pos < 0:
        break
    # 标准 PCI ROM：0x18 处存放 PCIR 结构偏移（小端 u16）
    if pos + 0x1C <= len(data):
        pcir_off = struct.unpack_from("<H", data, pos + 0x18)[0]
        sig = data[pos + pcir_off: pos + pcir_off + 4] if pcir_off < 0x100 else b""
        if sig == b"PCIR":
            vendor = struct.unpack_from("<H", data, pos + pcir_off + 4)[0]
            device = struct.unpack_from("<H", data, pos + pcir_off + 6)[0]
            rom_size = data[pos + 2] * 512
            cands.append((pos, rom_size, vendor, device, pcir_off))
    pos += 1

for pos, sz, vid, did, po in cands:
    tag = ""
    if vid == 0x10DE:
        tag = "  <== NVIDIA !!"
    print("  offset 0x%08X  size=%7d  VID=0x%04X DID=0x%04X pcir@0x%X%s"
          % (pos, sz, vid, did, po, tag))

# ---- 2. 找字符串 ----
print("\n[2] 关键字符串位置")
for needle in (b"NVIDIA", b"90.17.64", b"TU117", b"NPDS", b"Version"):
    hits = []
    p = 0
    while True:
        p = data.find(needle, p)
        if p < 0 or len(hits) >= 8:
            break
        hits.append(p)
        p += 1
    print("  %-10s : %d 处%s" % (needle.decode(errors="replace"), len(hits),
          "  " + ", ".join("0x%X" % h for h in hits[:6]) if hits else ""))

# ---- 3. 对 NVIDIA 候选做进一步解析 ----
nv = [c for c in cands if c[2] == 0x10DE]
print("\n[3] NVIDIA vBIOS 候选解析")
if not nv:
    print("  未找到 VID=0x10DE 的标准 PCI ROM，尝试定位 PCIR 暴力匹配")
    p = 0
    while True:
        p = data.find(b"PCIR", p)
        if p < 0:
            break
        if p + 12 <= len(data):
            vid = struct.unpack_from("<H", data, p + 4)[0]
            did = struct.unpack_from("<H", data, p + 6)[0]
            if vid == 0x10DE:
                print("  PCIR @0x%X  VID=0x10DE DID=0x%04X (ROM起点约 0x%X)"
                      % (p, did, max(0, p - 0x1A)))
        p += 1
else:
    for pos, sz, vid, did, po in nv:
        print("  起点 0x%08X  声明大小 %d" % (pos, sz))
        blob = data[pos:pos + max(sz, 0x10000)]
        for needle in (b"NVIDIA", b"TU117", b"90.17"):
            q = blob.find(needle)
            if q >= 0:
                print("    内含字符串 %-8s @ +0x%X" % (needle.decode(), q))
        # 打印 header 前 64 字节
        print("    header:", " ".join("%02X" % b for b in blob[:32]))

print("\n" + "=" * 70)
print("完成")
