# -*- coding: utf-8 -*-
"""精确测量 FlagGems 5.0.2 与 triton 2.3.1 的 API 缺口

方法:
  1. 静态扫描 flag_gems 全部 .py, 用正则提取所有 `tl.<attr>` / `triton.language.<attr>` 用法
  2. 与 triton 2.3.1 的 dir(triton.language) 求差集
  3. 把缺失项按出现次数排序 —— 这直接等于「要在 triton 2.3.1 上补多少个符号」

如果缺口小 -> 全部 monkey-patch 补上, FlagGems 可在现有环境跑
如果缺口大 -> 自己实现算子库
"""
import os, re, sys, collections

FG = "/vol1/ganlin/venv/lib/python3.11/site-packages/flag_gems"
if not os.path.isdir(FG):
    print("flag_gems 未安装:", FG)
    sys.exit(1)

# ---- 1. 现有 triton.language 属性
try:
    import triton, triton.language as tl
    have = set(dir(tl))
    print("triton 版本:", triton.__version__)
    print("triton.language 导出符号数:", len(have))
except Exception as e:
    print("import triton 失败:", e)
    have = set()
    tl = None

# ---- 2. 扫描 flag_gems 源码
PAT = re.compile(r"\btl\.([A-Za-z_][A-Za-z0-9_]*)")
PAT2 = re.compile(r"\btriton\.language\.([A-Za-z_][A-Za-z0-9_]*)")
PAT3 = re.compile(r"\btldevice\.([A-Za-z_][A-Za-z0-9_]*)")

used = collections.Counter()
files = 0
for root, dirs, names in os.walk(FG):
    for n in names:
        if not n.endswith(".py"):
            continue
        p = os.path.join(root, n)
        try:
            src = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        files += 1
        for m in PAT.finditer(src):
            used[m.group(1)] += 1
        for m in PAT2.finditer(src):
            used[m.group(1)] += 1
        for m in PAT3.finditer(src):
            used["(tldevice)" + m.group(1)] += 1

print("扫描 .py 文件数:", files)
print("出现的 tl.* 符号总数(去重):", len(used))
print()

# ---- 3. 求缺
missing = {k: v for k, v in used.items() if not k.startswith("(tldevice)") and k not in have}
present = {k: v for k, v in used.items() if not k.startswith("(tldevice)") and k in have}

print("=" * 74)
print("已有: %d 个" % len(present))
print("=" * 74)
for k, v in sorted(present.items(), key=lambda z: -z[1])[:14]:
    print("   %-26s x%d" % (k, v))

print()
print("=" * 74)
print("★ 缺失: %d 个（这就是要在 triton 2.3.1 上补的符号数）" % len(missing))
print("=" * 74)
for k, v in sorted(missing.items(), key=lambda z: -z[1]):
    print("   %-26s 被引用 x%d" % (k, v))

# ---- 4. 尝试从 tl.math / tl.extra.cuda 找替代
print()
print("=" * 74)
print("缺失符号的可替代来源探测")
print("=" * 74)
try:
    import triton.language.math as tmath
    import triton.language.extra.cuda as tcuda
    math_names = set(dir(tmath))
    cuda_names = set(dir(tcuda))
    fixed = 0
    for k in sorted(missing):
        srcs = []
        if k in math_names:
            srcs.append("tl.math")
        if k in cuda_names:
            srcs.append("tl.extra.cuda")
        if srcs:
            fixed += 1
            print("   %-26s -> %s 可提供" % (k, "/".join(srcs)))
        else:
            print("   %-26s -> !! 无处可取, 必须自己实现" % k)
    print()
    print("   可通过别名补齐: %d / %d" % (fixed, len(missing)))
    print("   必须自研: %d" % (len(missing) - fixed))
except Exception as e:
    print("  探测失败:", e)
