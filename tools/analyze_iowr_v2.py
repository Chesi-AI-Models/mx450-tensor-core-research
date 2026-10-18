#!/usr/bin/env python3
"""
analyze_iowr_v2.py - FECS 微码 iowr 地址提取（带寄存器值跟踪）

修正 v1 的 bug：v1 只跟踪 `mov`，不跟踪 `add`/`sub` 等算术运算，
导致 iowr 的实际地址被记成 mov 的初始值。

本版跟踪：
  mov $rN <imm>
  add b32 $rN $rN <imm>      / add $rN $rM <imm>
  sub b32 $rN $rN <imm>
  shl / shr / and / or
  mov $rN $rM

并记录每个 iowr 的：
  - 地址（跟踪后的值）
  - 是否可解析（未解析则标记 UNKNOWN）
  - 所在函数（用最近的 ret 之前的入口粗分）
"""

import re
import sys
from collections import Counter, defaultdict

ASM = sys.argv[1] if len(sys.argv) > 1 else "fecs_plain.asm"

# 指令正则
re_label   = re.compile(r"^([0-9a-f]{8}):")
re_mov_i   = re.compile(r"^\s*mov\s+\$r(\d+)\s+0x([0-9a-f]+)")
re_mov_r   = re.compile(r"^\s*mov\s+\$r(\d+)\s+\$r(\d+)")
re_add_i   = re.compile(r"^\s*add\s+(?:b32\s+)?\$r(\d+)\s+\$r(\d+)\s+0x([0-9a-f]+)")
re_sub_i   = re.compile(r"^\s*sub\s+(?:b32\s+)?\$r(\d+)\s+\$r(\d+)\s+0x([0-9a-f]+)")
re_shl     = re.compile(r"^\s*shl\s+(?:b32\s+)?\$r(\d+)\s+\$r(\d+)\s+0x([0-9a-f]+)")
re_shr     = re.compile(r"^\s*shr\s+(?:b32\s+)?\$r(\d+)\s+\$r(\d+)\s+0x([0-9a-f]+)")
re_and_i   = re.compile(r"^\s*and\s+(?:b32\s+)?\$r(\d+)\s+\$r(\d+)\s+0x([0-9a-f]+)")
re_or_i    = re.compile(r"^\s*or\s+(?:b32\s+)?\$r(\d+)\s+\$r(\d+)\s+0x([0-9a-f]+)")
re_iowr_r  = re.compile(r"iowr\s+I\[\$r(\d+)\]\s+\$r(\d+)")
re_iowr_i  = re.compile(r"iowr\s+I\[0x([0-9a-f]+)\]")
re_iord_r  = re.compile(r"iord\s+\$r(\d+)\s+I\[\$r(\d+)\]")
re_ret     = re.compile(r"\bret\s*$")
re_lcall   = re.compile(r"\blcall\s+0x([0-9a-f]+)")

cur = {}           # 寄存器 -> 已知常量
iowr_hits = []     # (addr or None, asm_line, func_start)
iord_hits = []
func_start = 0
pc = 0

lines = open(ASM, encoding="utf-8", errors="replace").read().splitlines()

for ln in lines:
    m = re_label.match(ln)
    if m:
        pc = int(m.group(1), 16)

    # --- 寄存器写入跟踪 ---
    m = re_mov_i.search(ln)
    if m:
        cur[m.group(1)] = int(m.group(2), 16)
        continue
    m = re_mov_r.search(ln)
    if m:
        if m.group(2) in cur:
            cur[m.group(1)] = cur[m.group(2)]
        else:
            cur.pop(m.group(1), None)
        continue
    m = re_add_i.search(ln)
    if m:
        dst, src, imm = m.group(1), m.group(2), int(m.group(3), 16)
        if src in cur:
            cur[dst] = (cur[src] + imm) & 0xFFFFFFFF
        else:
            cur.pop(dst, None)
        continue
    m = re_sub_i.search(ln)
    if m:
        dst, src, imm = m.group(1), m.group(2), int(m.group(3), 16)
        if src in cur:
            cur[dst] = (cur[src] - imm) & 0xFFFFFFFF
        else:
            cur.pop(dst, None)
        continue
    m = re_shl.search(ln)
    if m:
        dst, src, imm = m.group(1), m.group(2), int(m.group(3), 16)
        if src in cur:
            cur[dst] = (cur[src] << imm) & 0xFFFFFFFF
        else:
            cur.pop(dst, None)
        continue
    m = re_shr.search(ln)
    if m:
        dst, src, imm = m.group(1), m.group(2), int(m.group(3), 16)
        if src in cur:
            cur[dst] = (cur[src] >> imm) & 0xFFFFFFFF
        else:
            cur.pop(dst, None)
        continue
    m = re_and_i.search(ln)
    if m:
        dst, src, imm = m.group(1), m.group(2), int(m.group(3), 16)
        if src in cur:
            cur[dst] = (cur[src] & imm) & 0xFFFFFFFF
        else:
            cur.pop(dst, None)
        continue
    m = re_or_i.search(ln)
    if m:
        dst, src, imm = m.group(1), m.group(2), int(m.group(3), 16)
        if src in cur:
            cur[dst] = (cur[src] | imm) & 0xFFFFFFFF
        else:
            cur.pop(dst, None)
        continue

    # --- iowr 记录 ---
    m = re_iowr_r.search(ln)
    if m:
        reg = m.group(1)
        a = cur.get(reg)
        iowr_hits.append((a, ln.strip(), func_start, pc))
        continue
    m = re_iowr_i.search(ln)
    if m:
        iowr_hits.append((int(m.group(1), 16), ln.strip(), func_start, pc))
        continue

    # --- iord 记录 ---
    m = re_iord_r.search(ln)
    if m:
        reg = m.group(2)
        a = cur.get(reg)
        iord_hits.append((a, ln.strip(), func_start, pc))
        continue

    # --- 函数边界 ---
    m = re_lcall.search(ln)
    if m:
        func_start = int(m.group(1), 16)
        cur.clear()
        continue

# ============ 输出 ============
total = len(iowr_hits)
resolved = [h for h in iowr_hits if h[0] is not None]
unres = [h for h in iowr_hits if h[0] is None]

print("=" * 70)
print("iowr 总数: %d   可解析: %d (%.1f%%)   未解析: %d"
      % (total, len(resolved), 100.0 * len(resolved) / max(total, 1), len(unres)))
print("iord 总数: %d" % len(iord_hits))
print("=" * 70)

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

    print("\n=== 全部唯一地址（%d 个，升序）===" % len(cnt))
    addrs = sorted(cnt)
    for i in range(0, len(addrs), 6):
        row = addrs[i:i+6]
        print("  " + "  ".join("0x%06X(x%d)" % (a, cnt[a]) for a in row))

print("\n=== 未解析样本（前 15）===")
for a, ln, fs, pc in unres[:15]:
    print("  @%08x  %s" % (pc, ln[:70]))

print("\n=== 按函数分布（前 15 个函数）===")
fn = Counter(h[2] for h in resolved)
for f, n in fn.most_common(15):
    print("  func 0x%06X : %d 次 iowr" % (f, n))
