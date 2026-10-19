#!/usr/bin/env python3
"""
find_grctx_refs.py - 在 nvidia.ko 中定位引用 GRCTX_* 字符串的代码

原理：
  nvidia.ko 是 relocatable ELF（ET_REL）。
  .rodata 里的字符串被 .text 引用时，会产生 R_X86_64_PC32/PLT32 等重定位项，
  其 r_addend 指向 .rodata 的节内偏移（或符号值）。
  因此解析 .rela.text，筛出 addend 落在 GRCTX 字符串区间的项，
  即可得到「哪些代码位置引用了这些字符串」。

输出：
  1. GRCTX 字符串清单（偏移 -> 字符串）
  2. 每个字符串被哪些 .text 偏移引用
  3. 按被引用次数排序，指出最关键的布局描述
"""

import struct
import sys
import re
from collections import defaultdict

KO = sys.argv[1] if len(sys.argv) > 1 else \
    "/lib/modules/6.18.18.c1126-trim/updates/trim/alternatives/nvidia-gpu/nvidia.ko"

data = open(KO, "rb").read()
print("file size: %d (0x%X)" % (len(data), len(data)))

# ---------- ELF 头 ----------
assert data[:4] == b"\x7fELF", "not ELF"
is64 = data[4] == 2
e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
e_shstrndx = struct.unpack_from("<H", data, 0x3E)[0]

print("shoff=0x%X shentsize=%d shnum=%d shstrndx=%d" % (e_shoff, e_shentsize, e_shnum, e_shstrndx))

sections = []
for i in range(e_shnum):
    off = e_shoff + i * e_shentsize
    sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size = struct.unpack_from("<IIQQQQ", data, off)
    sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack_from("<IIQQ", data, off + 0x28)
    sections.append(dict(idx=i, name_off=sh_name, type=sh_type, flags=sh_flags,
                         addr=sh_addr, offset=sh_offset, size=sh_size,
                         link=sh_link, info=sh_info, align=sh_addralign, entsize=sh_entsize))

shstr = sections[e_shstrndx]
def sec_name(s):
    start = shstr["offset"] + s["name_off"]
    end = data.find(b"\0", start)
    return data[start:end].decode("ascii", "replace")

for s in sections:
    s["name"] = sec_name(s)

byname = {s["name"]: s for s in sections}
print("sections:", ", ".join(s["name"] for s in sections[:12]), "...")

text = byname[".text"]
rela_text = byname.get(".rela.text")
rodata = byname[".rodata"]

print("\n.text      off=0x%X size=0x%X" % (text["offset"], text["size"]))
print(".rela.text off=0x%X size=0x%X entsize=%d" % (rela_text["offset"], rela_text["size"], rela_text["entsize"]))
print(".rodata    off=0x%X size=0x%X" % (rodata["offset"], rodata["size"]))

# ---------- 提取 GRCTX 字符串（在 .rodata 内按偏移扫描） ----------
# 注意：strings -t x 给的是**文件偏移**，而 .rodata 的 sh_addr 通常为 0（ET_REL），
# 所以字符串的文件偏移 == 节内偏移（当 sh_addr==0 时）。这里统一用「文件偏移」比较，
# 重定位的 addend 在 ET_REL 下也是节内偏移；.rodata 的 sh_addr 为 0 时两者等价。
ro_base = rodata["offset"]
ro_end = ro_base + rodata["size"]
print("\n.rodata file range: 0x%X - 0x%X (sh_addr=0x%X)" % (ro_base, ro_end, rodata["addr"]))

grctx = {}   # file_off -> string
i = ro_base
while i < ro_end:
    j = data.find(b"\0", i)
    if j < 0:
        break
    s = data[i:j]
    if 6 <= len(s) <= 120 and s.startswith(b"GRCTX_"):
        # 重定位的 addend 是节内偏移；.rodata sh_addr=0 时节内偏移==虚拟地址
        grctx[i - ro_base] = s.decode("ascii", "replace")
    i = j + 1

print("找到 %d 个 GRCTX_ 字符串" % len(grctx))
if grctx:
    offs = sorted(grctx)
    print("节内范围: 0x%X - 0x%X (文件 0x%X - 0x%X)" % (offs[0], offs[-1], offs[0]+ro_base, offs[-1]+ro_base))

# ---------- 解析 .rela.text ----------
# Elf64_Rela: r_offset(8) r_info(8) r_addend(8)
# ET_REL 中 r_offset 是 .text 节内偏移（当 .text sh_addr==0 时）
R_X86_64 = {1: "R_X86_64_64", 2: "R_X86_64_PC32", 4: "R_X86_64_PLT32",
            9: "R_X86_64_GOTPCREL", 10: "R_X86_64_32", 11: "R_X86_64_32S"}

refs = defaultdict(list)   # grctx file_off -> [(text_off, reloc_type)]
type_count = defaultdict(int)

n = rela_text["size"] // 24
print("\n解析 %d 条 .rela.text 重定位..." % n)

lo, hi = (min(grctx), max(grctx)) if grctx else (0, 0)

for k in range(n):
    off = rela_text["offset"] + k * 24
    r_offset, r_info, r_addend = struct.unpack_from("<QQq", data, off)
    r_type = r_info & 0xFFFFFFFF
    if r_type not in R_X86_64:
        continue
    type_count[R_X86_64[r_type]] += 1
    if grctx and lo <= r_addend <= hi + 128:
        # 找最接近的字符串起点（addend 可能指向字符串中间）
        best = None
        for g in grctx:
            if g <= r_addend < g + 128:
                if best is None or g > best:
                    best = g
        if best is not None:
            refs[best].append((r_offset, R_X86_64[r_type]))

print("重定位类型分布:", dict(type_count))
print("\n被引用的 GRCTX 字符串: %d 个" % len(refs))

# ---------- 输出 ----------
if refs:
    print("\n" + "=" * 78)
    print("被引用最多的 GRCTX 字符串（前 40）")
    print("=" * 78)
    for g, lst in sorted(refs.items(), key=lambda kv: -len(kv[1]))[:40]:
        print("  %-46s x%-3d  首个引用 .text+0x%X" % (grctx[g], len(lst), lst[0][0]))

    print("\n" + "=" * 78)
    print("UCODE 相关（微码大小/地址常量最可能在此附近）")
    print("=" * 78)
    for g in sorted(grctx):
        if "UCODE" in grctx[g]:
            lst = refs.get(g, [])
            mark = ("  x%d  .text+0x%X" % (len(lst), lst[0][0])) if lst else "  (无直接引用)"
            print("  %-46s%s" % (grctx[g], mark))

    print("\n" + "=" * 78)
    print("REG_LIST 相关")
    print("=" * 78)
    for g in sorted(grctx):
        if "REG_LIST" in grctx[g] and "ADDR" in grctx[g]:
            lst = refs.get(g, [])
            mark = ("  x%d  .text+0x%X" % (len(lst), lst[0][0])) if lst else "  (无直接引用)"
            print("  %-46s%s" % (grctx[g], mark))
else:
    print("\n未找到直接引用。可能原因：")
    print("  1) .rodata 的 sh_addr 非 0，addend 需要减去 sh_addr")
    print("  2) 字符串通过 GOT/表间接引用")
    print("  3) 引用集中在 .rela.rodata 而非 .rela.text")
