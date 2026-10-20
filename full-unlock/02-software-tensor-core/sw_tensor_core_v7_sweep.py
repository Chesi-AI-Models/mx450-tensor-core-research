# -*- coding: utf-8 -*-
"""D1 第四步: tile 配置扫描 + 双缓冲（实测，不猜）

已知事实链（全部本机实测）:
  v2  标量 smem, BK=8 PAD=6, 128x128x256      1.603  (52.3%)
  v4  + 转置/BK扫描                            1.876  (61.3%)
  v5  + float4 全局加载 BK=16 PAD=8            1.9714 (64.4%)   <<< 当前最优
  v6  float smem + A转置 + LDS.128             1.811  (59.2%)   <<< 倒退，smem 字节翻倍
  诊断 TM8/TN8/256线程 -> 94 regs, 2 blocks/SM = 16 warps

本轮变量:
  1. tile: (BM,BN) = (128,128) (128,64) (64,64) (256,128)
  2. 线程数由 BM/TM * BN/TN 推出
  3. 双缓冲: 两个 smem buffer，先加载下一块再算当前块（T2-2 已证 1.14x 有效）
"""
import ctypes, glob, time
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
            "/usr/lib/x86_64-linux-gnu/libnvrtc.so*"])
        self.lib = ctypes.CDLL(path)

    def compile(self, src, arch="sm_75"):
        prog = ctypes.c_void_p()
        rc = self.lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"g.cu", 0, None, None)
        if rc:
            raise RuntimeError("create rc=%d" % rc)
        opts = [("--gpu-architecture=" + arch).encode(), b"--std=c++14"]
        arr = (ctypes.c_char_p * len(opts))(*opts)
        rc = self.lib.nvrtcCompileProgram(prog, len(opts), arr)
        if rc:
            sz = ctypes.c_size_t()
            self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
            buf = ctypes.create_string_buffer(sz.value + 1)
            self.lib.nvrtcGetProgramLog(prog, buf)
            raise RuntimeError("compile: " + buf.value.decode(errors="replace"))
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
        cu = self.cu
        cu.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                           ctypes.c_char_p]
        cu.cuModuleUnload.argtypes = [ctypes.c_void_p]
        cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        cu.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        cu.cuCtxSynchronize.argtypes = []
        cu.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                          ctypes.c_void_p]
        cu.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]

    def __getattr__(self, n):
        return getattr(self.cu, n)

    def load(self, ptx):
        mod = ctypes.c_void_p()
        rc = self.cu.cuModuleLoadData(ctypes.byref(mod), ptx.encode())
        if rc:
            raise RuntimeError("load rc=%d" % rc)
        fn = ctypes.c_void_p()
        rc = self.cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"gemm_ffma")
        if rc:
            raise RuntimeError("getfn rc=%d" % rc)
        return mod, fn

    def attrs(self, fn, nt):
        regs = ctypes.c_int()
        self.cu.cuFuncGetAttribute(ctypes.byref(regs), 4, fn)
        blocks = ctypes.c_int()
        self.cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(blocks), fn, nt, 0)
        return regs.value, blocks.value

    def malloc(self, n):
        p = ctypes.c_uint64()
        rc = self.cu.cuMemAlloc_v2(ctypes.byref(p), n)
        if rc:
            raise RuntimeError("malloc rc=%d" % rc)
        return p


SRC = r"""
#define BM __BM__
#define BN __BN__
#define BK __BK__
#define TM __TM__
#define TN __TN__
#define PAD __PAD__
#define NT __NT__
#define NX (BN / TN)          // 线程网格 x
#define NY (BM / TM)          // 线程网格 y

__device__ __forceinline__ float h2f(unsigned short h) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}

__device__ __forceinline__ void load_tile(const unsigned short* __restrict__ A,
                                          const unsigned short* __restrict__ B,
                                          unsigned short As[BM][BK + PAD],
                                          unsigned short Bs[BK][BN],
                                          int row0, int col0, int k0, int M, int N, int K)
{
    // A: 8 个 half = 16B 一组
    const int AV = BM * BK / 8;
    for (int t = 0; t < (AV + NT - 1) / NT; t++) {
        int v = threadIdx.x + t * NT;
        if (v >= AV) break;
        int r = v / (BK / 8);
        int c8 = (v % (BK / 8)) * 8;
        int gr = row0 + r, gc = k0 + c8;
        if (gr < M && gc + 7 < K) {
            *reinterpret_cast<uint4*>(&As[r][c8]) =
                *reinterpret_cast<const uint4*>(&A[gr * K + gc]);
        } else {
            for (int q = 0; q < 8; q++)
                As[r][c8 + q] = (gr < M && gc + q < K) ? A[gr * K + gc + q] : 0;
        }
    }
    // B
    const int BV = BK * BN / 8;
    for (int t = 0; t < (BV + NT - 1) / NT; t++) {
        int v = threadIdx.x + t * NT;
        if (v >= BV) break;
        int r = v / (BN / 8);
        int c8 = (v % (BN / 8)) * 8;
        int gr = k0 + r, gc = col0 + c8;
        if (gr < K && gc + 7 < N) {
            *reinterpret_cast<uint4*>(&Bs[r][c8]) =
                *reinterpret_cast<const uint4*>(&B[gr * N + gc]);
        } else {
            for (int q = 0; q < 8; q++)
                Bs[r][c8 + q] = (gr < K && gc + q < N) ? B[gr * N + gc + q] : 0;
        }
    }
}

__device__ __forceinline__ void do_mma(unsigned short As[BM][BK + PAD],
                                       unsigned short Bs[BK][BN],
                                       float acc[TM][TN], int tx, int ty)
{
    #pragma unroll
    for (int kk = 0; kk < BK; kk++) {
        float a[TM], b[TN];
        #pragma unroll
        for (int i = 0; i < TM; i++) a[i] = h2f(As[ty * TM + i][kk]);
        #pragma unroll
        for (int j = 0; j < TN; j++) b[j] = h2f(Bs[kk][tx * TN + j]);
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j++)
                acc[i][j] += a[i] * b[j];
    }
}

extern "C" __global__ void gemm_ffma(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ B,
    unsigned short* __restrict__ C,
    int M, int N, int K)
{
__SMDECL__
    const int tx = threadIdx.x % NX;
    const int ty = threadIdx.x / NX;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;
__BODY__
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gr = row0 + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gc = col0 + tx * TN + j;
            if (gr < M && gc < N) C[gr * N + gc] = f2h(acc[i][j]);
        }
    }
}
"""

BODY_SINGLE = r"""
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tile(A, B, As, Bs, row0, col0, k0, M, N, K);
        __syncthreads();
        do_mma(As, Bs, acc, tx, ty);
        __syncthreads();
    }
"""

BODY_DB = r"""
    load_tile(A, B, As0, Bs0, row0, col0, 0, M, N, K);
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += 2 * BK) {
        if (k0 + BK < K) load_tile(A, B, As1, Bs1, row0, col0, k0 + BK, M, N, K);
        do_mma(As0, Bs0, acc, tx, ty);
        __syncthreads();
        if (k0 + BK >= K) break;
        if (k0 + 2 * BK < K) load_tile(A, B, As0, Bs0, row0, col0, k0 + 2 * BK, M, N, K);
        do_mma(As1, Bs1, acc, tx, ty);
        __syncthreads();
    }
"""


def build(BM, BN, BK, TM, TN, PAD, db):
    nt = (BM // TM) * (BN // TN)
    if db:
        sm = ("    __shared__ unsigned short As0[BM][BK + PAD];\n"
              "    __shared__ unsigned short Bs0[BK][BN];\n"
              "    __shared__ unsigned short As1[BM][BK + PAD];\n"
              "    __shared__ unsigned short Bs1[BK][BN];")
        body = BODY_DB
    else:
        sm = ("    __shared__ unsigned short As[BM][BK + PAD];\n"
              "    __shared__ unsigned short Bs[BK][BN];")
        body = BODY_SINGLE
    src = SRC.replace("__SMDECL__", sm).replace("__BODY__", body)
    for k, v in (("__BM__", BM), ("__BN__", BN), ("__BK__", BK), ("__TM__", TM),
                 ("__TN__", TN), ("__PAD__", PAD), ("__NT__", nt)):
        src = src.replace(k, str(v))
    return src, nt


def main():
    print("=" * 96)
    print("D1 第四步: tile 配置扫描 + 双缓冲")
    print("=" * 96)
    torch.cuda.init()
    pr = torch.cuda.get_device_properties(0)
    print("  GPU: %s  SM=%d" % (pr.name, pr.multi_processor_count))

    N = 2048
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16)
    ref = torch.mm(a.float(), b.float())
    cu = Cuda()
    nv = Nvrtc()
    c_dev = cu.malloc(N * N * 2)

    print()
    print("  %-40s %8s %8s %7s %6s %6s %10s"
          % ("配置", "耗时ms", "TFLOPS", "效率", "regs", "blk", "max_err"))
    print("  " + "-" * 96)

    cases = [
        # label,                   BM,  BN,  BK, TM, TN, PAD, db
        ("S1 128x128x16 T8x8 P8 单缓冲", 128, 128, 16, 8, 8, 8, False),
        ("S2 128x64x16  T8x8 P8 单缓冲", 128, 64, 16, 8, 8, 8, False),
        ("S3 64x64x16   T8x8 P8 单缓冲", 64, 64, 16, 8, 8, 8, False),
        ("S4 256x128x16 T8x8 P8 单缓冲", 256, 128, 16, 8, 8, 8, False),
        ("S5 128x128x8  T8x8 P8 单缓冲", 128, 128, 8, 8, 8, 8, False),
        ("S6 128x128x16 T8x4 P8 单缓冲", 128, 128, 16, 8, 4, 8, False),
        ("S7 128x128x16 T4x4 P8 单缓冲", 128, 128, 16, 4, 4, 8, False),
        ("S8 128x128x16 T8x8 P8 双缓冲", 128, 128, 16, 8, 8, 8, True),
    ]

    results = []
    for label, BM, BN, BK, TM, TN, PAD, db in cases:
        try:
            src, nt = build(BM, BN, BK, TM, TN, PAD, db)
            if BM * BK % 8 or BK * BN % 8:
                print("  %-40s SKIP: tile 不能被 8 整除" % label)
                continue
            ptx = nv.compile(src)
            mod, fn = cu.load(ptx)
        except Exception as e:
            print("  %-40s FAIL: %s" % (label, str(e).replace("\n", " ")[:44]))
            continue
        regs, blk = cu.attrs(fn, nt)
        grid = ((N + BN - 1) // BN, (N + BM - 1) // BM, 1)
        kp = [ctypes.c_uint64(a.data_ptr()), ctypes.c_uint64(b.data_ptr()),
              ctypes.c_uint64(c_dev.value), ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(N)]
        pp = (ctypes.c_void_p * 6)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in kp])

        def launch():
            cu.cuLaunchKernel(fn, grid[0], grid[1], grid[2], nt, 1, 1, 0, None, pp, None)

        try:
            launch()
            cu.cuCtxSynchronize()
            out = torch.empty(N, N, device="cuda", dtype=torch.float16)
            cu.cuMemcpyDtoH_v2(ctypes.c_void_p(out.data_ptr()), c_dev.value, N * N * 2)
            err = (out.float() - ref).abs().max().item()
        except Exception as e:
            print("  %-40s RUN FAIL: %s" % (label, str(e)[:50]))
            cu.cuModuleUnload(mod)
            break
        if err > 1.0:
            print("  %-40s WRONG err=%.3f" % (label, err))
            cu.cuModuleUnload(mod)
            continue

        for _ in range(3):
            launch()
        cu.cuCtxSynchronize()
        t0 = time.perf_counter()
        it = 10
        for _ in range(it):
            launch()
        cu.cuCtxSynchronize()
        dt = (time.perf_counter() - t0) / it
        tf = 2.0 * N ** 3 / dt / 1e12
        print("  %-40s %8.3f %8.4f %6.1f%% %6d %6d %10.5f"
              % (label, dt * 1e3, tf, tf / 3.06 * 100, regs, blk, err))
        results.append((label, tf, regs, blk))
        cu.cuModuleUnload(mod)

    cu.cuMemFree_v2(c_dev)

    print()
    print("=" * 96)
    if results:
        best = max(results, key=lambda x: x[1])
        print("  最优: %-40s %.4f TFLOPS (%.1f%%)"
              % (best[0], best[1], best[1] / 3.06 * 100))
        print("  历史: 1.603(52.3%%) -> 1.876(61.3%%) -> 1.9714(64.4%%) -> %.4f(%.1f%%)"
              % (best[1], best[1] / 3.06 * 100))
        print("  目标 2.5 (85%%) 达成 %.0f%%；cuBLAS fp32 甜区 2.37 -> 差 %.0f%%"
              % (best[1] / 2.5 * 100, (1 - best[1] / 2.37) * 100))


if __name__ == "__main__":
    main()
