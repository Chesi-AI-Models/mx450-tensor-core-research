#!/usr/bin/env python3
"""
scan_rodata_grregs.py - 在 nvidia.ko 的 .rodata 中寻找 GR 寄存器地址表

思路：
  GR context 的寄存器列表（GRCTX_REG_LIST_*）最终会以「地址数组」形式
  硬编码在 .rodata 或 .data 中。FECS/GPCCS 微码访问的 0x20000-0x2FFFF 段
  如果属于 GR 内部寄存器，其地址表就应该出现在这里。

  扫描策略：
    1. 把 .rodata 按 4 字节小端切分
    2. 找落在 0x20000-0x3FFFF 的值
    3. 找出「连续 >= 4 个」这种值的区块（地址表特征）
    4. 输出区块位置、长度、前若干项

  另外扫描 0x400000-0x5FFFFF（PGRAPH 段）作为对照。
"""

import struct
import sys

KO = sys.argv[1] if len(sys.argv) > 1 else \
    "/lib/modules/6.18.18.c1126-trim/updates/trim/alternatives/nvidia-gpu/nvidia.ko"

data = open(KO, "rb").read()

e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
e_shstrndx = struct.unpack_from("<H", data, 0x3E)[0]

secs = []
for i in range(e_shnum):
    o = e_shoff + i * e_shentsize
    nm, ty, fl, ad, of, sz = struct.unpack_from("<IIQQQQ", data, o)
    lk, inf, al, es = struct.unpack_from("<IIQQ", data, o + 0x28)
    secs.append(dict(name_off=nm, type=ty, addr=ad, offset=of, size=sz, link=lk, entsize=es))

shstr = secs[e_shstrndx]
for s in secs:
    st = shstr["offset"] + s["name_off"]
    en = data.find(b"\0", st)
    s["name"] = data[st:en].decode("ascii", "replace")

byname = {s["name"]: s for s in secs}

def scan(section_name, lo, hi, min_run=4, label=""):
    s = byname.get(section_name)
    if not s:
        print("  [%s] 不存在" % section_name)
        return
    off, size = s["offset"], s["size"]
    n = size // 4
    print("\n=== 扫描 %s（%s），范围 0x%X-0x%X ===" % (section_name, label, lo, hi))
    print("    节大小 0x%X，%d 个 32 位值" % (size, n))

    runs = []
    cur = []
    for i in range(n):
        v = struct.unpack_from("<I", data, off + i * 4)[0]
        if lo <= v < hi:
            cur.append((i, v))
        else:
            if len(cur) >= min_run:
                runs.append(cur)
            cur = []
    if len(cur) >= min_run:
        runs.append(cur)

    print("    找到 %d 个连续区块（>=%d 项）" % (len(runs), min_run))
    # 按长度排序
    runs.sort(key=lambda r: -len(r))
    for r in runs[:12]:
        start_idx = r[0][0]
        addr = s["offset"] + start_idx * 4
        print("\n    --- 区块 @ 文件 0x%X（%d 项）---" % (addr, len(r)))
        vals = [v for _, v in r[:24]]
        for k in range(0, len(vals), 6):
            print("      " + "  ".join("0x%06X" % v for v in vals[k:k+6]))
        if len(r) > 24:
            print("      ... 共 %d 项" % len(r))
        # 检查是否等差
        if len(r) >= 3:
            diffs = [r[i+1][1] - r[i][1] for i in range(min(len(r)-1, 20))]
            uniq = set(diffs)
            if len(uniq) <= 3:
                print("      步长: %s" % ", ".join("0x%X" % d for d in sorted(uniq)))

print("=" * 78)
print("nvidia.ko .rodata GR 寄存器地址表扫描")
print("=" * 78)

scan(".rodata", 0x20000, 0x40000, 4, "GR 内部段 0x20000-0x3FFFF")
scan(".rodata", 0x400000, 0x600000, 4, "PGRAPH 段 0x400000-0x5FFFFF")
scan(".data", 0x20000, 0x40000, 4, "GR 内部段（.data）")
scan(".data", 0x400000, 0x600000, 4, "PGRAPH 段（.data）")

# 统计 .rodata 里 0x2xxxx 值的总数
s = byname[".rodata"]
n = s["size"] // 4
cnt2 = 0
cnt4 = 0
for i in range(n):
    v = struct.unpack_from("<I", data, s["offset"] + i * 4)[0]
    if 0x20000 <= v < 0x40000:
        cnt2 += 1
    elif 0x400000 <= v < 0x600000:
        cnt4 += 1
print("\n=== 总量统计 ===")
print("  .rodata 中 0x20000-0x3FFFF 的值: %d 个" % cnt2)
print("  .rodata 中 0x400000-0x5FFFFF 的值: %d 个" % cnt4)
