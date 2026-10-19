#!/usr/bin/env python3
"""
bench_fp16_vs_tc.py - 判定 MX450 的 mma 到底跑在什么单元上

假设：TU117 (MX450) 的 SM 里没有 Tensor Core，原本放 TC 的位置是 FP16 单元。
      如果成立，那么：
        - FP16 CUDA core 峰值 ≈ mma.sync 峰值
        - 两者都远低于"理论 TC 值"

方法：
  1. FP32 matmul（基线）
  2. FP16 matmul（走 CUDA core，非 TC）
  3. 对比之前的 mma.sync 实测 0.60 TFLOPS

注意：
  - 关闭 TF32，避免走 TC 路径
  - 用足够大的矩阵让计算主导
  - 多次迭代取稳定值
"""
import torch
import time
import sys

print("=" * 66)
print("MX450 FP16 vs FP32 峰值对比")
print("=" * 66)
print("GPU:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print("SM 数:", props.multi_processor_count)
print("显存: %.0f MiB" % (props.total_memory / 1048576))
print("CUDA cap: %d.%d" % (props.major, props.minor))
print()

# 关闭 TF32（TF32 会走 TC 路径，必须关）
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print("allow_tf32 =", torch.backends.cuda.matmul.allow_tf32)
print()


def bench(dtype, size, iters, warmup=3):
    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    flops = 2.0 * size ** 3 / dt
    del a, b, c
    torch.cuda.empty_cache()
    return flops / 1e12, dt


SIZE = 2048
ITERS = 20

print("矩阵规模 %d x %d，每次 %.1f GFLOP，迭代 %d 次" % (SIZE, SIZE, 2*SIZE**3/1e9, ITERS))
print("-" * 66)

results = {}
for name, dtype in [("FP32", torch.float32), ("FP16", torch.float16)]:
    try:
        tf, dt = bench(dtype, SIZE, ITERS)
        results[name] = tf
        print("  %-6s : %7.3f TFLOPS   (单次 %.1f ms)" % (name, tf, dt * 1000))
    except Exception as e:
        print("  %-6s : 失败 %s" % (name, e))

print("-" * 66)
if "FP32" in results and "FP16" in results:
    ratio = results["FP16"] / results["FP32"]
    print("  FP16 / FP32 = %.2fx" % ratio)

print()
print("=" * 66)
print("与之前 mma.sync 实测对比")
print("=" * 66)
print("  手写 PTX mma.sync  : 0.60 TFLOPS")
print("  cuBLAS TENSOR_OP   : 0.32 TFLOPS")
print()

if "FP16" in results:
    fp16 = results["FP16"]
    print("  本次 FP16 CUDA core: %.3f TFLOPS" % fp16)
    print()
    if abs(fp16 - 0.60) / max(fp16, 0.60) < 0.5:
        print("  >>> 判定: FP16 与 mma 性能同一量级")
        print("  >>> 结论: mma 被降级到 FP16 单元执行，硬件无 Tensor Core")
    elif fp16 > 0.60 * 2:
        print("  >>> 判定: FP16 明显快于 mma")
        print("  >>> 结论: mma 有独立的慢速路径，需进一步分析")
    else:
        print("  >>> 判定: 两者差异明显，需结合理论峰值分析")

print()
print("理论参考:")
print("  MX450 (896 CUDA @ ~1.4 GHz): FP32 峰值 ≈ %.1f TFLOPS" % (896 * 2 * 1.4 / 1000))
print("  若 TU117 有真 TC (RTX 2060 级别): FP16 TC 峰值 ≈ 6-12 TFLOPS")
