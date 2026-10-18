#!/usr/bin/env python3
"""
analyze_iowr_v3.py - FECS 微码 iowr 地址提取（带寄存器值跟踪）

v1 bug: 只跟踪 mov，不跟踪 add/sub -> 地址系统性错误
v2 bug: 正则用了 ^ 锚点，但反汇编行首是地址+字节码 -> 0% 解析

v3: 先剥离行首的 "地址: 字节码 " 前缀，再解析指令助记符。

跟踪的算术指令：mov(imm/reg), add, sub, shl, shr, and, or
记录每个 iowr/iord 的解析地址，未解析的单独统计。
"""

import re
import sys
from collections import Counter

ASM = sys.argv[1] if len(sys.argv) > 1 else "fecs_plain.asm"

# 行首: "00000076: 89 00 06 02           " -> 剥掉
PREFIX = re.compile(r"^[0-9a-fA-F]{8}:\s+(?:[0-9a-fA-F]{2}\s+)*")

def body(line):
    m = PREFIX.match(line)
    if m:
        return line[m.end():].rstrip()
    return line.rstrip()

# 指令正则（作用于 body 之后）
R_MOV_I  = re.compile(r"^[A-Z]?\s*mov\s+\$r(\d+)\s+0x([0-9a-fA-F]+)\s*$")
R_MOV_R  = re.compile(r"^[A-Z]?\s*mov\s+\$r(\d+)\s+\$r(\d+)\s*$")
R_ADD    = re.compile(r"^[A-Z]?\s*add\s+(?:b32\s+)?\$r(\d+)\s+\$r(\d+)\s+0x([0-9a-fA-F]+)")
R_SUB    = re.compile(r"^[A-Z]?\s*sub\s+(?:b32\s+)?\$r(\d+)\s+\$r(\d+)\s+0x([0-9a-fA-F]+)")
R_SHL    = re.compile(r"^[A-Z]?\s*shl\s+(?:b32\s+)?\$r(\d+)\s+\$r(\d+)\s+0x([0-9a-fA-F]+)")
R_SHR    = re.compile(r"^[A-Z]?\s*shr\s+(?:b32\s+)?\$r(\d+)\s+\$r(\d+)\s+0x([0-9a-fA-F]+)")
R_AND    = re.compile(r"^[A-Z]?\s*and\s+(?:b32\s+)?\$r(\d+)\s+\$r(\d+)\s+0x([0-9a-fA-F]+)")
R_OR     = re.compile(r"^[A-Z]?\s*or\s+(?:b32\s+)?\$r(\d+)\s+\$r(\d+)\s+0x([0-9a-fA-F]+)")
R_IOWR_R = re.compile(r"\biowr\s+I\[\$r(\d+)\]\s+\$r(\d+)")
R_IOWR_I = re.compile(r"\biowr\s+I\[0x([0-9a-fA-F]+)\]")
R_IORD_R = re.compile(r"\biord\s+\$r(\d+)\s+I\[\$r(\d+)\]")
R_IORD_I = re.compile(r"\biord\s+\$r(\d+)\s+I\[0x([0-9a-fA-F]+)\]")
R_LCALL  = re.compile(r"\blcall\s+0x([0-9a-fA-F]+)")

cur = {}
iowr_hits = []
iord_hits = []
func = 0
pc = 0
stats = Counter()

lines = open(ASM, encoding="utf-8", errors="replace").read().splitlines()

for raw in lines:
    m = re.match(r"^([0-9a-fA-F]{8}):", raw)
    if m:
        pc = int(m.group(1), 16)

    b = body(raw)
    if not b:
        continue

    # 函数边界
    m = R_LCALL.search(b)
    if m:
        func = int(m.group(1), 16)
        cur.clear()
        stats["lcall"] += 1
        continue

    # --- 写寄存器 ---
    m = R_MOV_I.match(b)
    if m:
        cur[m.group(1)] = int(m.group(2), 16)
        stats["mov_i"] += 1
        continue
    m = R_MOV_R.match(b)
    if m:
        if m.group(2) in cur:
            cur[m.group(1)] = cur[m.group(2)]
        else:
            cur.pop(m.group(1), None)
        stats["mov_r"] += 1
        continue
    for rx, op, name in ((R_ADD, "+", "add"), (R_SUB, "-", "sub"),
                         (R_SHL, "<<", "shl"), (R_SHR, ">>", "shr"),
                         (R_AND, "&", "and"), (R_OR, "|", "or")):
        m = rx.match(b)
        if m:
            dst, src, imm = m.group(1), m.group(2), int(m.group(3), 16)
            if src in cur:
                v = cur[src]
                if op == "+":   v = (v + imm) & 0xFFFFFFFF
                elif op == "-": v = (v - imm) & 0xFFFFFFFF
                elif op == "<<": v = (v << imm) & 0xFFFFFFFF
                elif op == ">>": v = (v >> imm) & 0xFFFFFFFF
                elif op == "&": v = v & imm
                elif op == "|": v = v | imm
                cur[dst] = v
            else:
                cur.pop(dst, None)
            stats[name] += 1
            break
    else:
        pass
    if any(rx.match(b) for rx in (R_ADD, R_SUB, R_SHL, R_SHR, R_AND, R_OR)):
        continue

    # --- iowr ---
    m = R_IOWR_R.search(b)
    if m:
        iowr_hits.append((cur.get(m.group(1)), raw.strip(), func, pc))
        continue
    m = R_IOWR_I.search(b)
    if m:
        iowr_hits.append((int(m.group(1), 16), raw.strip(), func, pc))
        continue

    # --- iord ---
    m = R_IORD_R.search(b)
    if m:
        iord_hits.append((cur.get(m.group(2)), raw.strip(), func, pc))
        continue
    m = R_IORD_I.search(b)
    if m:
        iord_hits.append((int(m.group(2), 16), raw.strip(), func, pc))
        continue

# ============ 输出 ============
total = len(iowr_hits)
resolved = [h for h in iowr_hits if h[0] is not None]
unres = [h for h in iowr_hits if h[0] is None]

print("=" * 72)
print("解析统计: %s" % dict(stats))
print("iowr 总数: %d   可解析: %d (%.1f%%)   未解析: %d"
      % (total, len(resolved), 100.0 * len(resolved) / max(total, 1), len(unres)))
print("iord 总数: %d   可解析: %d"
      % (len(iord_hits), len([h for h in iord_hits if h[0] is not None])))
print("=" * 72)

if resolved:
    cnt = Counter(h[0] for h in resolved)
    print("\n=== 访问最多的地址（前 30）===")
    for a, n in cnt.most_common(30):
        print("  0x%06X  x%d" % (a, n))

    print("\n=== 按 64KB 段归类 ===")
    seg = Counter((h[0] >> 16) << 16 for h in resolved)
    for s, n in sorted(seg.items()):
        print("  0x%06X-0x%06X : %4d 次 (%.1f%%)"
              % (s, s + 0xFFFF, n, 100.0 * n / len(resolved)))

    print("\n=== 唯一地址 %d 个（升序）===" % len(cnt))
    addrs = sorted(cnt)
    for i in range(0, len(addrs), 5):
        print("  " + "  ".join("0x%06X(x%d)" % (a, cnt[a]) for a in addrs[i:i+5]))

print("\n=== 未解析样本（前 12）===")
for a, ln, fs, p in unres[:12]:
    print("  @%08x  %s" % (p, ln[:76]))

print("\n=== 按函数分布（前 12）===")
for f, n in Counter(h[2] for h in resolved).most_common(12):
    print("  func 0x%06X : %d 次" % (f, n))
