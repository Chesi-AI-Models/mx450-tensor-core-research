# -*- coding: utf-8 -*-
"""B6: 用 triton23_compat 兼容层跑 FlagGems —— 实测能否工作 + 性能

这是「从零造轮子」准则的直接落地：
不因为有版本缺口就放弃，而是自己把缺口补齐，再看真实结果。
"""
import sys, time, traceback

sys.path.insert(0, "/tmp")

print("=" * 74)
print("[1] 应用 triton 2.3 -> 3.x 兼容层")
print("=" * 74)
try:
    import triton23_compat
    triton23_compat.report()
except Exception:
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 74)
print("[2] import torch / flag_gems")
print("=" * 74)
import torch
print("  torch:", torch.__version__, "cuda:", torch.cuda.is_available())
p = torch.cuda.get_device_properties(0)
print("  GPU:", p.name, "CC=%d.%d" % (p.major, p.minor), "SM=%d" % p.multi_processor_count)

fg = None
try:
    import flag_gems
    fg = flag_gems
    print("  flag_gems OK:", flag_gems.__version__)
    print("  device:", getattr(flag_gems, "device", "?"))
except Exception as e:
    print("  flag_gems FAIL:", type(e).__name__, str(e)[:300])
    traceback.print_exc()

if fg is None:
    print()
    print(">>> import 未通过, 到此为止")
    sys.exit(2)

# ---------------------------------------------------------------- 3 算子清单
print()
print("=" * 74)
print("[3] 可用算子清点")
print("=" * 74)
probe = ["mm", "bmm", "addmm", "softmax", "layer_norm", "rms_norm", "gelu",
         "silu", "relu", "add", "mul", "sub", "div", "sum", "mean", "var_mean",
         "argmax", "topk", "sort", "embedding", "cross_entropy_loss",
         "scaled_dot_product_attention", "flash_attention_forward", "conv2d"]
have = [x for x in probe if hasattr(fg, x)]
miss = [x for x in probe if not hasattr(fg, x)]
print("  有 (%d): %s" % (len(have), ", ".join(have)))
print("  无 (%d): %s" % (len(miss), ", ".join(miss)))
try:
    import flag_gems.ops as _ops
    print("  flag_gems.ops 导出符号数:", len([x for x in dir(_ops) if not x.startswith("_")]))
except Exception as e:
    print("  ops 统计失败:", str(e)[:100])

# ---------------------------------------------------------------- 4 实测
print()
print("=" * 74)
print("[4] 实测算子（功能 + 性能）")
print("=" * 74)

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

results = []
N = 512
a = torch.randn(N, N, device="cuda", dtype=torch.float32)
b = torch.randn(N, N, device="cuda", dtype=torch.float32)

# 4.1 mm
if hasattr(fg, "mm"):
    try:
        ref = torch.mm(a, b)
        got = fg.mm(a, b)
        torch.cuda.synchronize()
        err = (got - ref).abs().max().item()
        t_t = bench(lambda: torch.mm(a, b))
        t_g = bench(lambda: fg.mm(a, b))
        tf = 2.0 * N ** 3 / t_g / 1e12
        results.append(("mm", err, t_t, t_g))
        print("  mm        err=%.3e  torch=%7.3fms  gems=%7.3fms  %.2fx  %.4f TFLOPS"
              % (err, t_t * 1e3, t_g * 1e3, t_t / t_g, tf))
    except Exception as e:
        print("  mm        FAIL:", type(e).__name__, str(e)[:200])
else:
    print("  mm        无此算子")

# 4.2 小算子批量
X = torch.randn(256, 512, device="cuda", dtype=torch.float32)
W = torch.randn(512, device="cuda", dtype=torch.float32)
B = torch.randn(512, device="cuda", dtype=torch.float32)

def try_op(label, t_fn, g_fn, tol=1e-2):
    try:
        r = t_fn()
        g = g_fn()
        torch.cuda.synchronize()
        err = (g - r).abs().max().item()
        tt = bench(t_fn)
        tg = bench(g_fn)
        ok = "OK " if err <= tol else "ERR"
        results.append((label, err, tt, tg))
        print("  %-12s %s err=%.3e  torch=%7.3fms  gems=%7.3fms  %.2fx"
              % (label, ok, err, tt * 1e3, tg * 1e3, tt / tg))
    except Exception as e:
        print("  %-12s FAIL %s: %s" % (label, type(e).__name__, str(e)[:150]))

if hasattr(fg, "softmax"):
    try_op("softmax", lambda: torch.softmax(X, -1), lambda: fg.softmax(X, -1))
if hasattr(fg, "layer_norm"):
    try_op("layer_norm", lambda: torch.layer_norm(X, (512,), W, B),
           lambda: fg.layer_norm(X, (512,), W, B))
if hasattr(fg, "gelu"):
    try_op("gelu", lambda: torch.nn.functional.gelu(X), lambda: fg.gelu(X))
if hasattr(fg, "silu"):
    try_op("silu", lambda: torch.nn.functional.silu(X), lambda: fg.silu(X))
if hasattr(fg, "add"):
    try_op("add", lambda: X + X, lambda: fg.add(X, X))
if hasattr(fg, "sum"):
    try_op("sum", lambda: torch.sum(X, -1), lambda: fg.sum(X, -1))

# ---------------------------------------------------------------- 5 结论
print()
print("=" * 74)
print("[5] 结论")
print("=" * 74)
ok_n = sum(1 for _, e, _, _ in results if e <= 1e-2)
print("  成功算子: %d / %d" % (ok_n, len(results)))
if results:
    gem_faster = [r for r in results if r[3] < r[2]]
    print("  gems 更快的: %d 个" % len(gem_faster))
    for label, err, tt, tg in gem_faster:
        print("     %-12s %.2fx" % (label, tt / tg))
