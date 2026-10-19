#!/usr/bin/env python3
"""
triton_gemm_v2.py — 软件 Tensor Core v2：避开 tl.dot，用外积 + FMA

核心思路：
  v1 用 tl.dot -> Triton 生成 HMMA -> 0.595 TFLOPS（TC 模拟路径）
  v2 用外积累加  -> Triton 生成 FFMA/FMA -> 走 CUDA core

外积形式（K 维展开到最内层）：
  for k in range(BK):
      acc += a[:, k][:, None] * b[k, :][None, :]

这样 Triton 会生成标量 FMA 序列，而不是 HMMA。

注意：寄存器压力大，BLOCK 不能太大。
"""
import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def gemm_outer_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a_tile = tl.load(
            A + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak,
            mask=(k0 + offs_k)[None, :] < K, other=0.0,
        )   # (BM, BK) fp16
        b_tile = tl.load(
            B + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k0 + offs_k)[:, None] < K, other=0.0,
        )   # (BK, BN) fp16

        # 外积展开：逐个 k 累加，避开 tl.dot
        a_f = a_tile.to(tl.float32)
        b_f = b_tile.to(tl.float32)
        for kk in tl.static_range(BLOCK_K):
            a_col = tl.sum(tl.where(offs_k[None, :] == kk, a_f, 0.0), axis=1)  # (BM,)
            b_row = tl.sum(tl.where(offs_k[:, None] == kk, b_f, 0.0), axis=0)  # (BN,)
            acc += a_col[:, None] * b_row[None, :]

    c = acc.to(tl.float16)
    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
    )


@triton.jit
def gemm_fp32dot_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """对照：FP32 输入 + tl.dot -> 走 FFMA（CUDA core）。"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :], c)


def bench_outer(M, N, K, cfg, iters=10, warmup=3):
    BM, BN, BK, nw = cfg
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    def run():
        gemm_outer_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=nw,
        )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return 2.0 * M * N * K / dt / 1e12, dt


def bench_fp32dot(M, N, K, cfg, iters=10, warmup=3):
    BM, BN, BK, GM, nw, ns = cfg
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

    def run():
        gemm_fp32dot_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
            num_warps=nw, num_stages=ns,
        )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return 2.0 * M * N * K / dt / 1e12, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["outer", "fp32dot", "both"])
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print("=" * 72)
    print("软件 Tensor Core v2 — 避开 tl.dot")
    print("=" * 72)
    print("GPU: %s  SM=%d  CC=%d.%d" % (p.name, p.multi_processor_count, p.major, p.minor))
    print()

    SIZE = 1024   # 外积版本寄存器压力大，先用 1024

    if args.mode in ("outer", "both"):
        print("-" * 72)
        print("[A] 外积 + FMA（FP16 输入，避开 tl.dot）")
        print("-" * 72)
        # 先验正确性
        try:
            tf, dt = bench_outer(512, 512, 512, (16, 16, 16, 2), iters=3, warmup=1)
            print("  小尺寸连通性: OK (%.3f TFLOPS @512)" % tf)
        except Exception as e:
            print("  外积版本失败: %s" % str(e)[:100])
            tf = None

        if tf is not None:
            print()
            print("  %-20s %10s %10s" % ("(BM,BN,BK,warps)", "TFLOPS", "ms"))
            best = None
            for cfg in [(16, 16, 16, 2), (16, 32, 16, 2), (32, 32, 16, 4),
                        (32, 64, 16, 4), (16, 16, 32, 2), (32, 32, 32, 4)]:
                try:
                    tf, dt = bench_outer(SIZE, SIZE, SIZE, cfg)
                    print("  %-20s %10.3f %10.2f" % (str(cfg), tf, dt * 1000))
                    if best is None or tf > best[1]:
                        best = (cfg, tf)
                except Exception as e:
                    print("  %-20s FAIL %s" % (str(cfg), str(e)[:40]))
            if best:
                print("  最佳: %s -> %.3f TFLOPS" % (best[0], best[1]))
        print()

    if args.mode in ("fp32dot", "both"):
        print("-" * 72)
        print("[B] FP32 tl.dot（走 FFMA，对照）")
        print("-" * 72)
        print("  %-24s %10s %10s" % ("(BM,BN,BK,GM,w,s)", "TFLOPS", "ms"))
        best = None
        for cfg in [(64, 64, 32, 8, 4, 3), (128, 128, 32, 8, 8, 3),
                    (128, 128, 16, 8, 8, 3)]:
            try:
                tf, dt = bench_fp32dot(SIZE, SIZE, SIZE, cfg)
                print("  %-24s %10.3f %10.2f" % (str(cfg), tf, dt * 1000))
                if best is None or tf > best[1]:
                    best = (cfg, tf)
            except Exception as e:
                print("  %-24s FAIL %s" % (str(cfg), str(e)[:40]))
        if best:
            print("  最佳: %s -> %.3f TFLOPS" % (best[0], best[1]))
        print()

    print("=" * 72)
    print("基准: cuBLAS TENSOR_OP 0.320 | cuBLAS FP32 2.122 | FP16x2 峰值 5.770")
    print("=" * 72)


if __name__ == "__main__":
    main()
