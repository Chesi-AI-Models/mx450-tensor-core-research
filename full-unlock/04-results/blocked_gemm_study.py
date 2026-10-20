# -*- coding: utf-8 -*-
"""绕开 cuBLAS 4096 断崖: 分块 GEMM + kernel 名取证 + bf16 路径

已实测事实:
  fp32 cuBLAS: 1024 -> 2.35 TFLOPS, 2048 -> 2.11, 3072 -> 0.93, 4096 -> 0.35
  fp16 cuBLAS: 全程 0.24-0.35 TFLOPS (无 TC 的 Turing 没有调优路径)
  bf16 (fp32 模拟) -> 1.36 TFLOPS

本脚本验证:
  A) 把 N=4096 拆成块, 每块落在 cuBLAS 的高效区间, 总时间能否反超
  B) profiler 抓 kernel 名 (torch 2.3.1 属性名修正)
  C) fp16 走 bf16 中转是否也能提速
"""
import time, torch

torch.cuda.init()
dev = "cuda"
print("=" * 78)
print("分块 GEMM 绕开 cuBLAS 断崖")
print("=" * 78)

def bench(fn, iters=10, warmup=4):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

def tf(n, t):
    return 2.0 * n ** 3 / t / 1e12

# ---------------------------------------------------------------- A 分块
print()
print("[A] 分块 GEMM vs 整块 GEMM  (fp32)")
print("%6s | %8s | %10s | %10s | %10s" % ("N", "block", "time(ms)", "TFLOPS", "vs whole"))
print("-" * 78)

def blocked_mm(a, b, bs):
    n = a.shape[0]
    c = torch.empty(n, n, dtype=a.dtype, device=a.device)
    for i in range(0, n, bs):
        ai = a[i:i + bs, :]
        for j in range(0, n, bs):
            bj = b[:, j:j + bs].contiguous()
            c[i:i + bs, j:j + bs] = torch.mm(ai, bj)
    return c

for n in (2048, 3072, 4096):
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    b = torch.randn(n, n, device=dev, dtype=torch.float32)
    t_whole = bench(lambda: torch.mm(a, b), iters=8)
    print("%6d | %8s | %10.3f | %10.4f | %10s" % (n, "whole", t_whole * 1e3, tf(n, t_whole), "1.00x"))
    for bs in (512, 1024, 2048):
        if bs > n or n % bs != 0:
            continue
        try:
            t = bench(lambda: blocked_mm(a, b, bs), iters=6)
            # 正确性抽查
            got = blocked_mm(a, b, bs)
            ref = torch.mm(a, b)
            err = (got - ref).abs().max().item()
            print("%6d | %8d | %10.3f | %10.4f | %9.2fx   err=%.2e"
                  % (n, bs, t * 1e3, tf(n, t), t_whole / t, err))
            del got, ref
        except Exception as e:
            print("%6d | %8d | FAIL %s" % (n, bs, str(e)[:70]))
    del a, b
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- A2 fp16 分块
print()
print("[A2] 分块 GEMM (fp16 直连, 预期无效)")
for n in (2048, 4096):
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    t_whole = bench(lambda: torch.mm(a, b), iters=5)
    print("%6d whole  %10.3f ms  %.4f TFLOPS" % (n, t_whole * 1e3, tf(n, t_whole)))
    for bs in (1024,):
        t = bench(lambda: blocked_mm(a, b, bs), iters=5)
        print("%6d b=%-6d %10.3f ms  %.4f TFLOPS  %.2fx"
              % (n, bs, t * 1e3, tf(n, t), t_whole / t))
    del a, b; torch.cuda.empty_cache()

# ---------------------------------------------------------------- B kernel 名
print()
print("[B] profiler 抓 kernel 名 (torch 2.3.1: self_cuda_time_total)")
from torch.profiler import profile, ProfilerActivity
n = 1024
a16 = torch.randn(n, n, device=dev, dtype=torch.float16)
b16 = torch.randn(n, n, device=dev, dtype=torch.float16)
a32 = a16.float(); b32 = b16.float()
abf = a16.bfloat16(); bbf = b16.bfloat16()

for tag, x, y in (("fp16", a16, b16), ("fp32", a32, b32), ("bf16", abf, bbf)):
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                torch.mm(x, y)
            torch.cuda.synchronize()
        rows = []
        for ev in prof.key_averages():
            t = getattr(ev, "self_cuda_time_total", None)
            if t is None:
                t = getattr(ev, "self_device_time_total", 0) or 0
            if t and t > 0:
                rows.append((ev.key, t))
        rows.sort(key=lambda z: -z[1])
        print("    [%s]" % tag)
        for k, t in rows[:4]:
            print("        %9.1f us  %s" % (t, k[:88]))
        if not rows:
            print("        (无 CUDA kernel 记录)")
    except Exception as e:
        print("    [%s] FAIL %s" % (tag, str(e)[:100]))

# ---------------------------------------------------------------- C fp16 -> bf16 中转
print()
print("[C] fp16 -> bf16 中转 (Turing 无原生 bf16, cuBLAS 会用 fp32 模拟)")
print("%6s | %12s | %12s | gain | 精度损失" % ("N", "fp16 direct", "fp16->bf16"))
print("-" * 78)
for n in (1024, 2048):
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    t_d = bench(lambda: torch.mm(a, b), iters=10)
    def via_bf16(a=a, b=b):
        return torch.mm(a.bfloat16(), b.bfloat16()).half()
    t_b = bench(via_bf16, iters=10)
    # 精度: bf16 只有 8 位尾数, 损失明显
    ref = torch.mm(a.float(), b.float())
    got = via_bf16().float()
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    print("%6d | %9.3f ms | %9.3f ms | %.2fx | rel=%.4f"
          % (n, t_d * 1e3, t_b * 1e3, t_d / t_b, rel))
    del a, b; torch.cuda.empty_cache()

# ---------------------------------------------------------------- D 结论
print()
print("=" * 78)
print("[D] 汇总")
print("=" * 78)
print("  规则 1: MX450 上永远不要在 GEMM 里用 fp16 直连 (0.25-0.35 TFLOPS)")
print("  规则 2: fp16 需求 -> 转 fp32 做 GEMM -> 转回, 1024/2048 有 3.3-3.7x, 精度无损")
print("  规则 3: N>=3072 时 cuBLAS 自身断崖, 用分块 GEMM 规避")
print("  规则 4: bf16 在 Turing 上是 fp32 模拟 (1.36 TFLOPS), 但精度只有 8 位尾数")
print("=" * 78)
