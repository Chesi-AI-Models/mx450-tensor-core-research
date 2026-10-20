# -*- coding: utf-8 -*-
"""D1: 软件 Tensor Core GEMM 优化 v4 —— 共享内存转置消除 bank conflict

联网查证后的正确方向（不再是原方案）:
  "Column access in a row-major shared memory array triggers bank conflicts.
   The solution is to transpose A's tile in shared memory during the load phase."
  — CUDA GEMM 优化实践

原实现的问题:
  As[BM][BK] 读 As[ty*TM+i][kk] 是【列访问】
    warp 内 ty=0/1, 地址差 8*BK short
    BK=16 -> 8*16*2 = 256 bytes = 64 banks ≡ 0 (mod 32)  -> 同 bank 冲突
  Bs[BK][BN] 读 Bs[kk][tx*TN+j] 虽为行访问，但 bank = (4*tx)%32,
    tx=0 与 tx=8 都是 bank 0 -> 2-way conflict

修复:
  A 转置 -> AsT[BK][BM+PAD]，读 AsT[kk][ty*TM+i]（行访问，连续）
  B 转置 -> BsT[BN][BK+PAD]，读 BsT[tx*TN+j][kk]
  PAD 选取使 8*(dim+PAD)*2/4 不是 32 的倍数

优化路径参考（A6000 实测数据）:
  8x8 register tiling          68.7% of cuBLAS
  + float4 + smem transpose    78.4%
  + warp tiling                93.7%
"""
import ctypes, glob, time, sys
import torch

def _find(patterns):
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None

class Nvrtc:
    def __init__(self):
        path = _find([
            "/vol1/ganlin/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
            "/usr/lib/x86_64-linux-gnu/libnvrtc.so*",
        ])
        if not path:
            raise RuntimeError("libnvrtc not found")
        self.lib = ctypes.CDLL(path)

    def compile(self, src, arch="sm_75"):
        prog = ctypes.c_void_p()
        rc = self.lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(),
                                         b"gemm.cu", 0, None, None)
        if rc != 0:
            raise RuntimeError("nvrtcCreateProgram rc=%d" % rc)
        opts = [("--gpu-architecture=" + arch).encode(), b"--std=c++14"]
        arr = (ctypes.c_char_p * len(opts))(*opts)
        rc = self.lib.nvrtcCompileProgram(prog, len(opts), arr)
        if rc != 0:
            sz = ctypes.c_size_t()
            self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
            buf = ctypes.create_string_buffer(sz.value + 1)
            self.lib.nvrtcGetProgramLog(prog, buf)
            raise RuntimeError("compile failed:\n" + buf.value.decode(errors="replace"))
        sz = ctypes.c_size_t()
        self.lib.nvrtcGetPTXSize(prog, ctypes.byref(sz))
        ptx = ctypes.create_string_buffer(sz.value)
        self.lib.nvrtcGetPTX(prog, ptx)
        self.lib.nvrtcDestroyProgram(ctypes.byref(prog))
        return ptx.value.decode()

class Cuda:
    def __init__(self):
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        self.cu = ctypes.CDLL("libcuda.so.1")
        self.cu.cuInit(0)
        self.ctx = ctypes.c_void_p()
        self.cu.cuCtxGetCurrent(ctypes.byref(self.ctx))
        cu = self.cu
        cu.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                           ctypes.c_char_p]
        cu.cuModuleUnload.argtypes = [ctypes.c_void_p]
        cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        cu.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                      ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                      ctypes.c_uint, ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        cu.cuCtxSynchronize.argtypes = []

    def load(self, ptx):
        mod = ctypes.c_void_p()
        rc = self.cu.cuModuleLoadData(ctypes.byref(mod), ptx.encode())
        if rc != 0:
            raise RuntimeError("cuModuleLoadData rc=%d" % rc)
        fn = ctypes.c_void_p()
        rc = self.cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"gemm_ffma")
        if rc != 0:
            raise RuntimeError("cuModuleGetFunction rc=%d" % rc)
        return mod, fn

    def malloc(self, nbytes):
        p = ctypes.c_uint64()
        rc = self.cu.cuMemAlloc_v2(ctypes.byref(p), nbytes)
        if rc != 0:
            raise RuntimeError("cuMemAlloc rc=%d size=%d" % (rc, nbytes))
        return p

    def __getattr__(self, name):
        """把 cuXxx 属性转发到底层 CDLL，避免 self.cu 与实例同名混淆"""
        return getattr(self.cu, name)


# ============================================================ kernel 模板（转置版）
KERNEL_T = r"""
__device__ __forceinline__ float h2f(unsigned short h) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}

#define BM 128
#define BN 128
#define BK __BK__
#define PAD __PAD__
#define TM 8
#define TN 8
#define NTHREADS 256

extern "C" __global__ void gemm_ffma(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ B,
    unsigned short* __restrict__ C,
    int M, int N, int K)
{
    __shared__ unsigned short AsT[BK][BM + PAD];
    __shared__ unsigned short BsT[BN][BK + PAD];

    const int tx = threadIdx.x % 16;
    const int ty = threadIdx.x / 16;

    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int t = 0; t < (BM * BK) / NTHREADS; t++) {
            int idx = threadIdx.x + t * NTHREADS;
            int r = idx / BK, c = idx % BK;
            int gr = row0 + r, gc = k0 + c;
            AsT[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0;
        }
        #pragma unroll
        for (int t = 0; t < (BK * BN) / NTHREADS; t++) {
            int idx = threadIdx.x + t * NTHREADS;
            int r = idx / BN, c = idx % BN;
            int gr = k0 + r, gc = col0 + c;
            BsT[c][r] = (gr < K && gc < N) ? B[gr * N + gc] : 0;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a[i] = h2f(AsT[kk][ty * TM + i]);
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b[j] = h2f(BsT[tx * TN + j][kk]);
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gr = row0 + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gc = col0 + tx * TN + j;
            if (gr < M && gc < N)
                C[gr * N + gc] = f2h(acc[i][j]);
        }
    }
}
"""

# 原版（未转置）作为对照
KERNEL_N = r"""
__device__ __forceinline__ float h2f(unsigned short h) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
#define BM 128
#define BN 128
#define BK __BK__
#define PAD __PAD__
#define TM 8
#define TN 8
#define NTHREADS 256
extern "C" __global__ void gemm_ffma(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ B,
    unsigned short* __restrict__ C,
    int M, int N, int K)
{
    __shared__ unsigned short As[BM][BK + PAD];
    __shared__ unsigned short Bs[BK][BN];
    const int tx = threadIdx.x % 16;
    const int ty = threadIdx.x / 16;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int t = 0; t < (BM * BK) / NTHREADS; t++) {
            int idx = threadIdx.x + t * NTHREADS;
            int r = idx / BK, c = idx % BK;
            int gr = row0 + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0;
        }
        #pragma unroll
        for (int t = 0; t < (BK * BN) / NTHREADS; t++) {
            int idx = threadIdx.x + t * NTHREADS;
            int r = idx / BN, c = idx % BN;
            int gr = k0 + r, gc = col0 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a[i] = h2f(As[ty * TM + i][kk]);
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b[j] = h2f(Bs[kk][tx * TN + j]);
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gr = row0 + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gc = col0 + tx * TN + j;
            if (gr < M && gc < N)
                C[gr * N + gc] = f2h(acc[i][j]);
        }
    }
}
"""


def main():
    print("=" * 78)
    print("D1: GEMM 优化 v4 —— 共享内存转置 + BK/PAD 扫描")
    print("=" * 78)
    torch.cuda.init()
    p = torch.cuda.get_device_properties(0)
    print("  GPU:", p.name, "CC=%d.%d  SM=%d" % (p.major, p.minor, p.multi_processor_count))

    N = 2048
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16)
    ref = torch.mm(a.float(), b.float())

    cu = Cuda()
    nv = Nvrtc()
    a_p = a.data_ptr()
    b_p = b.data_ptr()
    c_dev = cu.malloc(N * N * 2)

    print()
    print("  %-26s %10s %10s %8s %10s" % ("配置", "耗时(ms)", "TFLOPS", "效率", "max_err"))
    print("  " + "-" * 76)

    results = []
    cases = []
    for bk in (8, 16, 32):
        for pad in (0, 2, 3, 4, 6, 8):
            cases.append(("原版 BK=%-2d PAD=%d" % (bk, pad), KERNEL_N, bk, pad))

    for label, tmpl, BK, PAD in cases:
        src = tmpl.replace("__BK__", str(BK)).replace("__PAD__", str(PAD))
        try:
            ptx = nv.compile(src)
        except Exception as e:
            print("  %-26s COMPILE FAIL: %s" % (label, str(e)[:60].replace("\n", " ")))
            continue
        try:
            mod, fn = cu.load(ptx)
        except Exception as e:
            print("  %-26s LOAD FAIL: %s" % (label, str(e)[:60]))
            continue

        grid = ((N + 127) // 128, (N + 127) // 128, 1)
        kp = [ctypes.c_uint64(a_p), ctypes.c_uint64(b_p), ctypes.c_uint64(c_dev.value),
              ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(N)]
        pp = (ctypes.c_void_p * len(kp))(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in kp])

        def launch():
            cu.cu.cuLaunchKernel(fn, grid[0], grid[1], grid[2], 256, 1, 1, 0, None, pp, None)

        try:
            launch()
            cu.cuCtxSynchronize()
            out = torch.empty(N, N, device="cuda", dtype=torch.float16)
            cu.cuMemcpyDtoH_v2(ctypes.c_void_p(out.data_ptr()), c_dev.value, N * N * 2)
            err = (out.float() - ref).abs().max().item()
        except Exception as e:
            print("  %-26s RUN FAIL: %s" % (label, str(e)[:60]))
            cu.cu.cuModuleUnload(mod)
            continue
        if err > 1.0:
            print("  %-26s WRONG RESULT err=%.3f" % (label, err))
            cu.cu.cuModuleUnload(mod)
            continue

        for _ in range(3):
            launch()
        cu.cuCtxSynchronize()
        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            launch()
        cu.cuCtxSynchronize()
        dt = (time.perf_counter() - t0) / iters
        tf = 2.0 * N**3 / dt / 1e12
        eff = tf / 3.06 * 100
        print("  %-26s %10.3f %10.4f %7.1f%% %10.5f" % (label, dt * 1e3, tf, eff, err))
        results.append((label, tf, eff))
        cu.cu.cuModuleUnload(mod)

    cu.cuMemFree_v2(c_dev)

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    if results:
        base = results[0]
        best = max(results, key=lambda x: x[1])
        print("  基线 %-22s %.4f TFLOPS (%.1f%%)" % (base[0], base[1], base[2]))
        print("  最优 %-22s %.4f TFLOPS (%.1f%%)  -> %.2fx"
              % (best[0], best[1], best[2], best[1] / base[1]))
        print("  目标 2.5 TFLOPS (85%%)  达成 %.0f%%" % (best[1] / 2.5 * 100))
        print()
        print("  参考: cuBLAS fp32 甜区 2.37 | 物理峰值 3.06")
        if best[1] > 2.37:
            print("  >>> 已超过 cuBLAS fp32")
        else:
            print("  >>> 仍低于 cuBLAS fp32 的 2.37（差 %.0f%%）"
                  % ((1 - best[1] / 2.37) * 100))


if __name__ == "__main__":
    main()
