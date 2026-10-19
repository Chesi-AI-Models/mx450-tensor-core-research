#!/usr/bin/env python3
"""
sw_tensor_core.py — 软件 Tensor Core：用 FP16x2 FMA 在 CUDA core 上重建矩阵乘

背景：
  TU117（MX450）没有 Tensor Core 硬件。HMMA 指令由 FP16 单元模拟，
  实测 0.60 TFLOPS（手写 mma.sync）/ 0.32（cuBLAS TENSOR_OP）。
  但 FP16x2 打包 FMA 的纯计算峰值是 5.77 TFLOPS。

  本文件用 NVRTC 编译一段 CUDA C（__half2 + __hfma2），
  在 CUDA core 上实现 GEMM，绕开 HMMA。

技术栈：
  CUDA C 源码 -> libnvrtc (NVRTC) 编译 -> PTX -> libcuda (cuModuleLoadData) -> 运行

用法：
  python sw_tensor_core.py [--size 2048] [--tile 8]
"""
import argparse
import ctypes
import os
import time
import glob

import torch


# ============================================================
# CUDA C 源码：FP16x2 分块 GEMM
# ============================================================
CUDA_SRC = r"""
#include <cuda_fp16.h>

// 每个 block: 128 线程 (4x32)，每个线程计算 TILE x TILE 输出
// A: M x K (row-major), B: K x N (row-major), C: M x N
// 使用 __half2 做 FP16x2 FMA

#ifndef TILE
#define TILE 8
#endif
#define BM 64
#define BN 64
#define BK 16

extern "C" __global__ void gemm_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    __shared__ __half As[BM][BK];
    __shared__ __half Bs[BK][BN];

    // 线程映射：16x16 = 256 线程，每个算 4x4（BM/16=4, BN/16=4）
    const int tx = threadIdx.x % 16;
    const int ty = threadIdx.x / 16;

    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 加载 A tile: 64x16 = 1024 元素 / 256 线程 = 4 个/线程
        #pragma unroll
        for (int t = 0; t < (BM * BK) / 256; t++) {
            int idx = threadIdx.x + t * 256;
            int r = idx / BK;
            int c = idx % BK;
            int gr = row0 + r;
            int gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * lda + gc] : __float2half(0.0f);
        }
        // 加载 B tile: 16x64 = 1024 元素 / 256 线程 = 4 个/线程
        #pragma unroll
        for (int t = 0; t < (BK * BN) / 256; t++) {
            int idx = threadIdx.x + t * 256;
            int r = idx / BN;
            int c = idx % BN;
            int gr = k0 + r;
            int gc = col0 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * ldb + gc] : __float2half(0.0f);
        }
        __syncthreads();

        // 计算：每个线程 4x4
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                a[i] = __half2float(As[ty * 4 + i][kk]);
            #pragma unroll
            for (int j = 0; j < 4; j++)
                b[j] = __half2float(Bs[kk][tx * 4 + j]);
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    // 写回
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int gr = row0 + ty * 4 + i;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int gc = col0 + tx * 4 + j;
            if (gr < M && gc < N)
                C[gr * ldc + gc] = __float2half(acc[i][j]);
        }
    }
}

// ---------- 版本 2：真正的 __half2 打包 FMA ----------
// 每个线程算 8x8，用 __hfma2 一次做两个 FP16 乘加
extern "C" __global__ void gemm_h2(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    __shared__ __half As[BM][BK];
    __shared__ __half Bs[BK][BN];

    const int tx = threadIdx.x % 16;
    const int ty = threadIdx.x / 16;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    // 每个线程 4x4 个 __half2 = 8x8 标量
    __half2 acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j] = __floats2half2_rn(0.0f, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int t = 0; t < (BM * BK) / 256; t++) {
            int idx = threadIdx.x + t * 256;
            int r = idx / BK, c = idx % BK;
            int gr = row0 + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        #pragma unroll
        for (int t = 0; t < (BK * BN) / 256; t++) {
            int idx = threadIdx.x + t * 256;
            int r = idx / BN, c = idx % BN;
            int gr = k0 + r, gc = col0 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            __half2 a2[4], b2[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                // 取 As[ty*8 + i*2 .. +1] 两个相邻 half 组成 half2
                a2[i] = *reinterpret_cast<const __half2*>(&As[ty * 8 + i * 2][kk]);
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                // Bs[kk][tx*8 + j*2 .. +1]
                b2[j] = *reinterpret_cast<const __half2*>(&Bs[kk][tx * 8 + j * 2]);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] = __hfma2(a2[i], b2[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int gr = row0 + ty * 8 + i * 2;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int gc = col0 + tx * 8 + j * 2;
            if (gr + 1 < M && gc + 1 < N)
                *reinterpret_cast<__half2*>(&C[gr * N + gc]) = acc[i][j];
        }
    }
}
"""


# ============================================================
# NVRTC + CUDA Driver 封装
# ============================================================
def _find_lib(patterns):
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


class Nvrtc:
    def __init__(self):
        path = _find_lib([
            "/vol1/ganlin/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
            "/usr/lib/x86_64-linux-gnu/libnvrtc.so*",
        ])
        if not path:
            raise RuntimeError("libnvrtc not found")
        self.lib = ctypes.CDLL(path)
        self.lib.nvrtcCreateProgram.restype = ctypes.c_int
        self.lib.nvrtcCompileProgram.restype = ctypes.c_int

    def compile(self, src, arch="sm_75"):
        prog = ctypes.c_void_p()
        rc = self.lib.nvrtcCreateProgram(
            ctypes.byref(prog), src.encode(), b"gemm.cu", 0, None, None)
        if rc != 0:
            raise RuntimeError("nvrtcCreateProgram rc=%d" % rc)

        # 头文件搜索路径
        inc = _find_lib([
            "/vol1/ganlin/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/include",
        ])
        opts = [
            ("--gpu-architecture=" + arch).encode(),
            b"--std=c++14",
            b"-default-device",
        ]
        if inc:
            opts.append(("--include-path=" + inc).encode())

        arr = (ctypes.c_char_p * len(opts))(*opts)
        rc = self.lib.nvrtcCompileProgram(prog, len(opts), arr)
        if rc != 0:
            # 取日志
            lsz = ctypes.c_size_t()
            self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(lsz))
            buf = ctypes.create_string_buffer(lsz.value + 1)
            self.lib.nvrtcGetProgramLog(prog, buf)
            raise RuntimeError("nvrtcCompileProgram failed:\n" + buf.value.decode(errors="replace"))

        psz = ctypes.c_size_t()
        self.lib.nvrtcGetPTXSize(prog, ctypes.byref(psz))
        ptx = ctypes.create_string_buffer(psz.value)
        self.lib.nvrtcGetPTX(prog, ptx)
        self.lib.nvrtcDestroyProgram(ctypes.byref(prog))
        return ptx.value.decode()


class Cuda:
    def __init__(self):
        self.cu = ctypes.CDLL("libcuda.so.1")
        self.cu.cuInit(0)
        dev = ctypes.c_int()
        self.cu.cuDeviceGet(ctypes.byref(dev), 0)
        self.dev = dev
        self.ctx = ctypes.c_void_p()
        self.cu.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev)
        self.cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self.cu.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        self.cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        self.mods = []

    def load(self, ptx):
        m = ctypes.c_void_p()
        rc = self.cu.cuModuleLoadData(ctypes.byref(m), ptx.encode() + b"\x00")
        if rc != 0:
            raise RuntimeError("cuModuleLoadData rc=%d" % rc)
        self.mods.append(m)
        return m

    def func(self, mod, name):
        f = ctypes.c_void_p()
        rc = self.cu.cuModuleGetFunction(ctypes.byref(f), mod, name.encode())
        if rc != 0:
            raise RuntimeError("cuModuleGetFunction(%s) rc=%d" % (name, rc))
        return f

    def alloc(self, nbytes):
        p = ctypes.c_uint64()
        rc = self.cu.cuMemAlloc_v2(ctypes.byref(p), nbytes)
        if rc != 0:
            raise RuntimeError("cuMemAlloc rc=%d" % rc)
        return p

    def h2d(self, dst, src_ptr, nbytes):
        self.cu.cuMemcpyHtoD_v2(dst, src_ptr, nbytes)

    def d2h(self, src, dst_ptr, nbytes):
        self.cu.cuMemcpyDtoH_v2(dst_ptr, src, nbytes)

    def launch(self, f, gx, gy, gz, bx, by, bz, args):
        arr = (ctypes.c_void_p * len(args))()
        keep = []
        for i, a in enumerate(args):
            if isinstance(a, ctypes.c_uint64):
                keep.append(a)
                arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
            else:
                keep.append(a)
                arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
        rc = self.cu.cuLaunchKernel(f, gx, gy, gz, bx, by, bz, 0, None, arr, None)
        if rc != 0:
            raise RuntimeError("cuLaunchKernel rc=%d" % rc)
        self.cu.cuCtxSynchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--which", default="h2", choices=["f16", "h2"])
    args = ap.parse_args()

    print("=" * 74)
    print("软件 Tensor Core — FP16x2 FMA on CUDA core (NVRTC 编译)")
    print("=" * 74)
    p = torch.cuda.get_device_properties(0)
    print("GPU: %s  SM=%d  CC=%d.%d" % (p.name, p.multi_processor_count, p.major, p.minor))
    print()

    print("[1] 编译 CUDA C ...")
    nv = Nvrtc()
    ptx = nv.compile(CUDA_SRC, "sm_75")
    print("    PTX 长度: %d 字符" % len(ptx))
    has_hfma2 = "fma.rn.f16x2" in ptx or "fma.rn.f16x2" in ptx
    has_hmma = "mma.sync" in ptx
    print("    含 fma.rn.f16x2: %s   含 mma.sync: %s" % (has_hfma2, has_hmma))
    with open("/tmp/sw_tc.ptx", "w") as f:
        f.write(ptx)
    print("    PTX 已存 /tmp/sw_tc.ptx")
    print()

    print("[2] 加载到 GPU ...")
    cu = Cuda()
    mod = cu.load(ptx)
    fname = "gemm_f16" if args.which == "f16" else "gemm_h2"
    fn = cu.func(mod, fname)
    print("    kernel: %s" % fname)
    print()

    M = N = K = args.size
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)

    dA = cu.alloc(a.numel() * 2)
    dB = cu.alloc(b.numel() * 2)
    dC = cu.alloc(c.numel() * 2)
    cu.h2d(dA, a.data_ptr(), a.numel() * 2)
    cu.h2d(dB, b.data_ptr(), b.numel() * 2)

    BM, BN = 64, 64
    gx, gy = (N + BN - 1) // BN, (M + BM - 1) // BM
    args_list = [ctypes.c_uint64(dA.value), ctypes.c_uint64(dB.value),
                 ctypes.c_uint64(dC.value),
                 ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]

    print("[3] 运行 ...")
    cu.launch(fn, gx, gy, 1, 256, 1, 1, args_list)
    cu.d2h(dC, c.data_ptr(), c.numel() * 2)

    ref = (a.float() @ b.float()).half()
    err = (c.float() - ref.float()).abs().max().item()
    print("    max abs err = %.5f" % err)
    print()

    print("[4] 测速 ...")
    iters = 20
    t0 = time.time()
    for _ in range(iters):
        cu.launch(fn, gx, gy, 1, 256, 1, 1, args_list)
    cu.cu.cuCtxSynchronize()
    dt = (time.time() - t0) / iters
    tf = 2.0 * M * N * K / dt / 1e12
    print("    %.3f TFLOPS   (%.1f ms)" % (tf, dt * 1000))
    print()
    print("=" * 74)
    print("对比:")
    print("  cuBLAS TENSOR_OP : 0.320 TFLOPS   -> %.1fx" % (tf / 0.32))
    print("  cuBLAS FP32      : 2.122 TFLOPS   -> %.2fx" % (tf / 2.122))
    print("  FP16x2 纯计算峰值 : 5.770 TFLOPS   -> %.2fx" % (tf / 5.770))
    print("=" * 74)


if __name__ == "__main__":
    main()
