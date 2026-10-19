#!/usr/bin/env python3
"""
triton_gemm.py — 软件 Tensor Core 第一版：Triton FP16 GEMM

目标：在 TU117（无 TC 硬件）上，用 CUDA core 的 FP16 路径
      跑出远高于 cuBLAS TENSOR_OP（0.32 TFLOPS）的性能。

对比基准（本机实测）：
  cuBLAS TENSOR_OP (cutlass_70_tensorop)  : 0.32 TFLOPS
  cuBLAS FP32 (volta_sgemm)               : 2.12 TFLOPS
  PTX FP16x2 纯计算峰值                    : 5.77 TFLOPS
  PTX mma.sync                            : 0.60 TFLOPS

用法：
  python triton_gemm.py [--check]
  --check 会额外 dump 生成的 PTX，检查是否含 HMMA
"""
import argparse
import time
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def gemm_fp16_kernel(
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
    """标准分块 GEMM，FP16 输入 / FP32 累加 / FP16 输出。"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache 友好的分组重排
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
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


def bench(M, N, K, config, iters=20, warmup=5):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)

    BM, BN, BK, GM, nw, ns = config
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

    def run():
        gemm_fp16_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
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
    flops = 2.0 * M * N * K
    return flops / dt / 1e12, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="dump PTX 检查 HMMA")
    args = ap.parse_args()

    print("=" * 72)
    print("软件 Tensor Core v1 — Triton FP16 GEMM on TU117")
    print("=" * 72)
    p = torch.cuda.get_device_properties(0)
    print("GPU: %s  SM=%d  CC=%d.%d" % (p.name, p.multi_processor_count, p.major, p.minor))
    print()

    # 正确性检查
    M = N = K = 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    cfg = (64, 64, 32, 8, 4, 3)
    grid = (triton.cdiv(M, cfg[0]) * triton.cdiv(N, cfg[1]),)
    gemm_fp16_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=cfg[0], BLOCK_N=cfg[1], BLOCK_K=cfg[2], GROUP_M=cfg[3],
        num_warps=cfg[4], num_stages=cfg[5],
    )
    ref = (a.float() @ b.float()).half()
    err = (c.float() - ref.float()).abs().max().item()
    print("正确性: max abs err = %.5f  (512x512x512)" % err)
    print()

    if args.check:
        # 从编译缓存里捞 PTX
        try:
            ptx = gemm_fp16_kernel.cache[torch.cuda.current_device()][0]["ptx"]
            has_hmma = "hmma" in ptx.lower() or "mma.sync" in ptx.lower()
            has_fma2 = "fma.rn.f16x2" in ptx
            print("PTX 检查: 含 HMMA/mma.sync = %s ; 含 fma.rn.f16x2 = %s" % (has_hmma, has_fma2))
            with open("/tmp/triton_gemm.ptx", "w") as f:
                f.write(ptx)
            print("PTX 已存 /tmp/triton_gemm.ptx")
        except Exception as e:
            print("PTX dump 失败:", e)
        print()

    # 扫描配置
    print("-" * 72)
    print("%-22s %10s %10s" % ("config (BM,BN,BK,GM,w,s)", "TFLOPS", "ms"))
    print("-" * 72)

    configs = [
        (64, 64, 32, 8, 4, 3),
        (64, 128, 32, 8, 4, 3),
        (128, 64, 32, 8, 4, 3),
        (128, 128, 32, 8, 8, 3),
        (128, 128, 64, 8, 8, 3),
        (128, 256, 32, 8, 8, 3),
        (256, 128, 32, 8, 8, 3),
        (64, 64, 64, 8, 4, 4),
        (128, 128, 32, 8, 8, 4),
    ]

    SIZE = 2048
    best = None
    for cfg in configs:
        try:
            tf, dt = bench(SIZE, SIZE, SIZE, cfg)
            print("%-22s %10.3f %10.2f" % (str(cfg), tf, dt * 1000))
            if best is None or tf > best[1]:
                best = (cfg, tf)
        except Exception as e:
            print("%-22s  FAIL  %s" % (str(cfg), str(e)[:40]))

    print("-" * 72)
    if best:
        print("最佳: %s -> %.3f TFLOPS" % (best[0], best[1]))
        print()
        print("对比:")
        print("  cuBLAS TENSOR_OP : 0.320 TFLOPS")
        print("  cuBLAS FP32      : 2.122 TFLOPS")
        print("  FP16x2 纯计算峰值 : 5.770 TFLOPS")
        print("  加速比 vs cuBLAS TENSOR_OP: %.1fx" % (best[1] / 0.32))


if __name__ == "__main__":
    main()
