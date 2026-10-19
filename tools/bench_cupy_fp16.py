#!/usr/bin/env python3
"""
bench_cupy_fp16.py - 用 CuPy 直接调 cuBLAS 测 FP16/FP32 峰值

目的：排除 PyTorch kernel 选择的影响，直接看 cuBLAS 的裸性能。
      如果 CuPy 的 FP16 也是 ~0.3 TFLOPS，则确认是硬件限制。
"""
import time
import sys

try:
    import cupy as cp
except ImportError:
    print("CuPy 未安装，尝试 pip install cupy-cuda12x")
    sys.exit(1)

print("=" * 66)
print("CuPy (cuBLAS 直调) FP16 vs FP32 峰值")
print("=" * 66)
print("CuPy:", cp.__version__)
dev = cp.cuda.Device(0)
props = cp.cuda.runtime.getDeviceProperties(0)
print("GPU:", props["name"].decode())
print("SM 数:", props["multiProcessorCount"])
print()


def bench(dtype, size, iters, warmup=3):
    a = cp.random.randn(size, size, dtype=dtype)
    b = cp.random.randn(size, size, dtype=dtype)
    for _ in range(warmup):
        c = cp.matmul(a, b)
    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    for _ in range(iters):
        c = cp.matmul(a, b)
    cp.cuda.Stream.null.synchronize()
    dt = (time.time() - t0) / iters
    flops = 2.0 * size ** 3 / dt
    del a, b, c
    cp.get_default_memory_pool().free_all_blocks()
    return flops / 1e12, dt


SIZE = 2048
ITERS = 20
print("矩阵 %d x %d，每次 %.1f GFLOP，迭代 %d" % (SIZE, SIZE, 2*SIZE**3/1e9, ITERS))
print("-" * 66)

res = {}
for name, dt_ in [("float32", cp.float32), ("float16", cp.float16), ("float64", cp.float64)]:
    try:
        tf, dt = bench(dt_, SIZE, ITERS)
        res[name] = tf
        print("  %-8s : %7.3f TFLOPS  (%.1f ms)" % (name, tf, dt * 1000))
    except Exception as e:
        print("  %-8s : 失败 %s" % (name, e))

print("-" * 66)
print()
print("判读标准:")
print("  - 若 float16 ≈ float32 或更快  -> 有原生 FP16 硬件")
print("  - 若 float16 << float32        -> FP16 走模拟/低效路径")
print("  - 若存在真 Tensor Core，float16 应达 4-12 TFLOPS")
