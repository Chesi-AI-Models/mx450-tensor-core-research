#!/usr/bin/env python3
"""
triton_gemm_v3.py — 软件 Tensor Core v3：FP32 dot（走 FFMA / CUDA core）

v1 (tl.dot, FP16) -> 0.595 TFLOPS  （Triton 生成 HMMA，TC 模拟路径）
v2 (外积)         -> 0.122 TFLOPS  （tl.where+tl.sum 太低效）
v3 (tl.dot, FP32) -> 目标 2.1 TFLOPS（走 FFMA，绕开 HMMA）

关键：输入 FP32、input_precision="ieee"，Triton 会生成 FFMA 而非 HMMA/TF32。
      FP32 的理论峰值是 2.5 TFLOPS，做到 2.1 就是 84% 效率。
      对比 cuBLAS TENSOR_OP 的 0.32，是 6.6 倍。

简化：去掉 GROUP_M 分组（v2 的编译错误来源），用 2D grid。
"""
import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if ACC_DTYPE == tl.float32:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)


def bench(dtype, M, N, K, cfg, iters=15, warmup=3):
    BM, BN, BK, nw, ns = cfg
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    acc_dtype = tl.float32 if dtype == torch.float32 else tl.float32

    def run():
        gemm_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            ACC_DTYPE=acc_dtype,
            num_warps=nw, num_stages=ns,
        )

    run()
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).half()
    err = (c.float() - ref.float()).abs().max().item()

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return 2.0 * M * N * K / dt / 1e12, dt, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048)
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print("=" * 74)
    print("软件 Tensor Core v3 — FP32 dot（FFMA 路径）")
    print("=" * 74)
    print("GPU: %s  SM=%d  CC=%d.%d" % (p.name, p.multi_processor_count, p.major, p.minor))
    print("矩阵 %d x %d x %d" % (args.size, args.size, args.size))
    print()

    configs = [
        (64, 64, 32, 4, 3),
        (64, 128, 32, 4, 3),
        (128, 64, 32, 4, 3),
        (128, 128, 32, 8, 3),
        (128, 128, 16, 8, 3),
        (64, 64, 64, 4, 3),
        (128, 256, 32, 8, 3),
    ]

    for label, dtype in [("FP32 输入 (FFMA)", torch.float32),
                         ("FP16 输入 (HMMA)", torch.float16)]:
        print("-" * 74)
        print(label)
        print("-" * 74)
        print("  %-22s %10s %10s %10s" % ("(BM,BN,BK,warps,stages)", "TFLOPS", "ms", "maxerr"))
        best = None
        for cfg in configs:
            try:
                tf, dt, err = bench(dtype, args.size, args.size, args.size, cfg)
                print("  %-22s %10.3f %10.2f %10.4f" % (str(cfg), tf, dt * 1000, err))
                if best is None or tf > best[1]:
                    best = (cfg, tf)
            except Exception as e:
                print("  %-22s FAIL %s" % (str(cfg), str(e).replace("\n", " ")[:44]))
        if best:
            print("  最佳: %s -> %.3f TFLOPS" % (best[0], best[1]))
        print()

    print("=" * 74)
    print("基准: cuBLAS TENSOR_OP 0.320 | cuBLAS FP32 2.122 | FP16x2 峰值 5.770")
    print("=" * 74)


if __name__ == "__main__":
    main()
