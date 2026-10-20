# -*- coding: utf-8 -*-
"""cuBLAS 路径深度研究: 为什么 MX450 上 FP16 GEMM 比 FP32 慢 9.5x, 以及怎么绕过去

假设 (三方资料 + 初步实测):
  cuBLAS 在无 Tensor Core 的 Turing (TU117) 上没有调优过的 "非 tensor-core FP16"
  GEMM 路径, 被迫走 HMMA 模拟 / 未调优 kernel, 实测 ~0.25 TFLOPS。
  而 FP32 SGEMM 正常, ~2.36 TFLOPS。

待验证的解法:
  A) fp16 -> fp32 -> SGEMM -> fp16  (转换开销 vs 9.5x 速度差)
  B) 不同尺寸下 cuBLAS heuristic 的行为 (4096 是否有断崖)
  C) 抓 kernel 名, 确认到底走了哪条路径
  D) 与自研软件 Tensor Core (1.603 TFLOPS) 对比
"""
import time, torch

torch.cuda.init()
dev = "cuda"
p = torch.cuda.get_device_properties(0)
print("=" * 78)
print("cuBLAS path study on %s  CC=%d.%d  SM=%d" % (p.name, p.major, p.minor, p.multi_processor_count))
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("=" * 78)

def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

def tflops(n, t):
    return 2.0 * n ** 3 / t / 1e12

# ---------------------------------------------------------------- 1. 尺寸扫描
print()
print("[1] 尺寸扫描: fp16 vs fp32 (纯 cuBLAS)")
print("%6s | %10s %10s | %10s %10s | ratio" % ("N", "fp16 TF", "fp16 ms", "fp32 TF", "fp32 ms"))
print("-" * 78)
scan = {}
for n in (256, 512, 1024, 2048, 3072, 4096):
    a16 = torch.randn(n, n, device=dev, dtype=torch.float16)
    b16 = torch.randn(n, n, device=dev, dtype=torch.float16)
    a32 = a16.float()
    b32 = b16.float()
    try:
        t16 = bench(lambda: torch.mm(a16, b16))
    except Exception as e:
        print("%6d | fp16 FAIL %s" % (n, str(e)[:40])); t16 = None
    try:
        t32 = bench(lambda: torch.mm(a32, b32))
    except Exception as e:
        print("%6d | fp32 FAIL %s" % (n, str(e)[:40])); t32 = None
    if t16 and t32:
        r = tflops(n, t32) / tflops(n, t16)
        scan[n] = (tflops(n, t16), tflops(n, t32))
        print("%6d | %10.4f %10.3f | %10.4f %10.3f | fp32/fp16 = %.2fx"
              % (n, tflops(n, t16), t16 * 1e3, tflops(n, t32), t32 * 1e3, r))
    del a16, b16, a32, b32
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- 2. bf16
print()
print("[2] bf16 (Turing 无原生 bf16, 预期更差)")
for n in (1024, 2048):
    try:
        a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        t = bench(lambda: torch.mm(a, b))
        print("    N=%-5d %10.3f ms -> %.4f TFLOPS" % (n, t * 1e3, tflops(n, t)))
        del a, b; torch.cuda.empty_cache()
    except Exception as e:
        print("    N=%-5d FAIL %s" % (n, str(e)[:80]))

# ---------------------------------------------------------------- 3. 混合路径
print()
print("[3] 混合路径: fp16 -> fp32 -> SGEMM -> fp16  (绕开坏掉的 fp16 GEMM)")
print("%6s | %12s | %12s | %12s | gain" % ("N", "fp16 direct", "fp16->32->16", "fp32 pure"))
print("-" * 78)

def mixed_mm(a16, b16):
    """把 fp16 转成 fp32 做 GEMM 再转回 fp16"""
    c32 = torch.mm(a16.float(), b16.float())
    return c32.half()

for n in (512, 1024, 2048, 3072, 4096):
    a16 = torch.randn(n, n, device=dev, dtype=torch.float16)
    b16 = torch.randn(n, n, device=dev, dtype=torch.float16)
    try:
        t_direct = bench(lambda: torch.mm(a16, b16))
        t_mixed = bench(lambda: mixed_mm(a16, b16), iters=15)
        t_pure = bench(lambda: torch.mm(a16.float(), b16.float()), iters=15)
        g = t_direct / t_mixed
        print("%6d | %8.3f ms | %8.3f ms | %8.3f ms | %.2fx" %
              (n, t_direct * 1e3, t_mixed * 1e3, t_pure * 1e3, g))
        # 精度
        ref = torch.mm(a16.float(), b16.float()).half()
        got = mixed_mm(a16, b16)
        err = (got.float() - ref.float()).abs().max().item()
        print("        精度: max_abs_err = %.6f (fp16 舍入量级)" % err)
    except Exception as e:
        print("%6d | FAIL %s" % (n, str(e)[:90]))
    del a16, b16; torch.cuda.empty_cache()

# ---------------------------------------------------------------- 4. kernel 名
print()
print("[4] profiler 抓 kernel 名 (确认 cuBLAS 到底走了哪条路径)")
n = 1024
a16 = torch.randn(n, n, device=dev, dtype=torch.float16)
b16 = torch.randn(n, n, device=dev, dtype=torch.float16)
a32 = a16.float(); b32 = b16.float()

for tag, x, y in (("fp16", a16, b16), ("fp32", a32, b32)):
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                torch.mm(x, y)
            torch.cuda.synchronize()
        names = []
        for ev in prof.key_averages():
            if ev.device_type is not None and "cuda" in str(ev.device_type).lower():
                if ev.self_device_time_total > 0:
                    names.append((ev.key, ev.self_device_time_total))
        names.sort(key=lambda z: -z[1])
        print("    %s top kernels:" % tag)
        for k, t in names[:5]:
            print("        %8.1f us  %s" % (t, k[:90]))
    except Exception as e:
        print("    %s profiler FAIL: %s" % (tag, str(e)[:90]))

# ---------------------------------------------------------------- 5. 数学模式开关
print()
print("[5] 各种后端开关对 fp16 mm 的影响 (N=1024)")
n = 1024
a16 = torch.randn(n, n, device=dev, dtype=torch.float16)
b16 = torch.randn(n, n, device=dev, dtype=torch.float16)
base = bench(lambda: torch.mm(a16, b16))
print("    baseline                 %8.3f ms  %.4f TFLOPS" % (base * 1e3, tflops(n, base)))

try:
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    t = bench(lambda: torch.mm(a16, b16))
    print("    no_fp16_reduced_reduce   %8.3f ms  %.4f TFLOPS  (%.2fx)" % (t * 1e3, tflops(n, t), base / t))
except Exception as e:
    print("    reduced_precision 开关失败:", str(e)[:60])

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    t = bench(lambda: torch.mm(a16, b16))
    print("    allow_tf32=False         %8.3f ms  %.4f TFLOPS  (%.2fx)" % (t * 1e3, tflops(n, t), base / t))
except Exception as e:
    print("    tf32 开关失败:", str(e)[:60])

# ---------------------------------------------------------------- 6. 结论
print()
print("=" * 78)
print("[6] 结论")
print("=" * 78)
if 1024 in scan:
    f16, f32 = scan[1024]
    print("    N=1024: fp16 %.4f TFLOPS | fp32 %.4f TFLOPS | FP32 快 %.1fx" % (f16, f32, f32 / f16))
print("    自研软件 Tensor Core (历史实测) = 1.603 TFLOPS")
if 1024 in scan:
    print("    自研 / cuBLAS-fp16 = %.1fx" % (1.603 / scan[1024][0]))
print("=" * 78)
