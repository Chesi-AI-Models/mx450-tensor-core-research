# -*- coding: utf-8 -*-
"""D1 测量校准器 —— 先解决「同一个配置测出 1.89 和 2.31 两个数」的问题

问题: A2 与上一轮 S5 参数完全相同(128x128xBK8 T8x8 PAD8 单缓冲)，
      却给出 1.8927 和 2.3075。MX450 在 NAS 里受散热/TGP 限制会掉频。

做法:
  1. 每轮开始和结束读 nvidia-smi 的 clocks.sm / 温度 / 功耗
  2. **交错测量**: 每轮把所有配置都测一遍，重复 R 轮，取中位数
  3. **校准点**: 每轮都测一次 cuBLAS fp32 mm 和 fp16 mm，用它们的漂移量
     判断整轮是否可信；并给出「相对 cuBLAS fp32」的归一化比值
"""
import ctypes, glob, subprocess, statistics, time
import torch


def _find(patterns):
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


def smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,"
                            "clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout.strip()
        return out.replace("\n", " | ")
    except Exception as e:
        return "smi failed: %s" % e


class Nvrtc:
    def __init__(self):
        self.lib = ctypes.CDLL(_find([
            "/vol1/ganlin/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
            "/usr/lib/x86_64-linux-gnu/libnvrtc.so*"]))

    def compile(self, src, arch="sm_75"):
        prog = ctypes.c_void_p()
        if self.lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"g.cu", 0, None, None):
            raise RuntimeError("create")
        opts = [("--gpu-architecture=" + arch).encode(), b"--std=c++14"]
        arr = (ctypes.c_char_p * len(opts))(*opts)
        if self.lib.nvrtcCompileProgram(prog, len(opts), arr):
            sz = ctypes.c_size_t()
            self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
            buf = ctypes.create_string_buffer(sz.value + 1)
            self.lib.nvrtcGetProgramLog(prog, buf)
            raise RuntimeError(buf.value.decode(errors="replace"))
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
        self.cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"gemm_ffma")
        return mod, fn

    def attrs(self, fn, nt):
        r = ctypes.c_int()
        self.cu.cuFuncGetAttribute(ctypes.byref(r), 4, fn)
        b = ctypes.c_int()
        self.cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(b), fn, nt, 0)
        return r.value, b.value

    def malloc(self, n):
        p = ctypes.c_uint64()
        if self.cu.cuMemAlloc_v2(ctypes.byref(p), n):
            raise RuntimeError("malloc")
        return p


SRC = r"""
#define BM __BM__
#define BN __BN__
#define BK __BK__
#define TM __TM__
#define TN __TN__
#define PAD __PAD__
#define NT __NT__
#define NX (BN / TN)

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
    const int AV = BM * BK / 8;
    for (int t = 0; t < (AV + NT - 1) / NT; t++) {
        int v = threadIdx.x + t * NT;
        if (v >= AV) break;
        int r = v / (BK / 8);
        int c8 = (v % (BK / 8)) * 8;
        int gr = row0 + r, gc = k0 + c8;
        if (gr < M && gc + 7 < K)
            *reinterpret_cast<uint4*>(&As[r][c8]) = *reinterpret_cast<const uint4*>(&A[gr * K + gc]);
        else
            for (int q = 0; q < 8; q++)
                As[r][c8 + q] = (gr < M && gc + q < K) ? A[gr * K + gc + q] : 0;
    }
    const int BV = BK * BN / 8;
    for (int t = 0; t < (BV + NT - 1) / NT; t++) {
        int v = threadIdx.x + t * NT;
        if (v >= BV) break;
        int r = v / (BN / 8);
        int c8 = (v % (BN / 8)) * 8;
        int gr = k0 + r, gc = col0 + c8;
        if (gr < K && gc + 7 < N)
            *reinterpret_cast<uint4*>(&Bs[r][c8]) = *reinterpret_cast<const uint4*>(&B[gr * N + gc]);
        else
            for (int q = 0; q < 8; q++)
                Bs[r][c8 + q] = (gr < K && gc + q < N) ? B[gr * N + gc + q] : 0;
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
            for (int j = 0; j < TN; j++) acc[i][j] += a[i] * b[j];
    }
}

extern "C" __global__ void gemm_ffma(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ B,
    unsigned short* __restrict__ C, int M, int N, int K)
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
    print("=" * 100)
    print("D1 测量校准器（交错 + 中位数 + cuBLAS 校准点）")
    print("=" * 100)
    torch.cuda.init()
    print("  起始状态:", smi())

    N = 2048
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16)
    ah, bh = a, b
    af, bf = a.float(), b.float()
    ref = torch.mm(af, bf)

    cu = Cuda()
    nv = Nvrtc()
    c_dev = cu.malloc(N * N * 2)

CASES = [
    ("C1 128x128x8  P8  单", 128, 128, 8, 8, 8, 8, False),
    ("C2 128x128x8  P8  双", 128, 128, 8, 8, 8, 8, True),
    ("C3 128x128x16 P8  单", 128, 128, 16, 8, 8, 8, False),
    ("C4 128x128x16 P8  双", 128, 128, 16, 8, 8, 8, True),
    ("C5 128x64x16  P8  单", 128, 64, 16, 8, 8, 8, False),
    ("C6 128x64x16  P8  双", 128, 64, 16, 8, 8, 8, True),
    ("C7 64x128x8   P0  单", 64, 128, 8, 8, 8, 0, False),
    ("C8 128x128x8  P16 双", 128, 128, 8, 8, 8, 16, True),
]


def run_one(idx, rounds=5):
    """单进程只跑一个配置，避免任何 context 污染；同时测 cuBLAS 校准点"""
    label, BM, BN, BK, TM, TN, PAD, db = CASES[idx]
    N = 2048
    torch.cuda.init()
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16)
    af, bf = a.float(), b.float()
    ref = torch.mm(af, bf)

    cu, nv = Cuda(), Nvrtc()
    c_dev = cu.malloc(N * N * 2)
    src, nt = build(BM, BN, BK, TM, TN, PAD, db)
    mod, fn = cu.load(nv.compile(src))
    regs, blk = cu.attrs(fn, nt)
    grid = ((N + BN - 1) // BN, (N + BM - 1) // BM, 1)
    kp = [ctypes.c_uint64(a.data_ptr()), ctypes.c_uint64(b.data_ptr()),
          ctypes.c_uint64(c_dev.value), ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(N)]
    pp = (ctypes.c_void_p * 6)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in kp])

    def launch():
        return cu.cuLaunchKernel(fn, grid[0], grid[1], grid[2], nt, 1, 1, 0, None, pp, None)

    rc = launch()
    if rc:
        return "RESULT %d | BUILD_OK LAUNCH_RC=%d" % (idx, rc)
    rc = cu.cuCtxSynchronize()
    if rc:
        return "RESULT %d | LAUNCH_RC=0 SYNC_RC=%d (kernel 自身崩)" % (idx, rc)
    out = torch.empty(N, N, device="cuda", dtype=torch.float16)
    cu.cuMemcpyDtoH_v2(ctypes.c_void_p(out.data_ptr()), c_dev.value, N * N * 2)
    err = (out.float() - ref).abs().max().item()
    if err > 1.0:
        return "RESULT %d | WRONG err=%.3f" % (idx, err)

    def timeit(f, it=5):
        f()
        cu.cuCtxSynchronize()
        t0 = time.perf_counter()
        for _ in range(it):
            f()
        cu.cuCtxSynchronize()
        return (time.perf_counter() - t0) / it

    def l32():
        torch.mm(af, bf, out=ref)

    def l16():
        torch.mm(a, b)

    ts, c32, c16 = [], [], []
    for _ in range(rounds):
        c32.append(timeit(l32))
        c16.append(timeit(l16))
        ts.append(timeit(launch))
    flop = 2.0 * N ** 3
    return ("RESULT %d | %-22s | best %.4f med %.4f TF | regs %d blk %d | "
            "cublas32 %.4f cublas16 %.4f TF | drift %.2fx"
            % (idx, label, flop / min(ts) / 1e12,
               flop / statistics.median(ts) / 1e12, regs, blk,
               flop / min(c32) / 1e12, flop / min(c16) / 1e12,
               max(c32) / min(c32)))


def main():
    import sys
    print("=" * 108)
    print("D1 校准 benchmark —— 每个配置独立进程，cuBLAS 作同进程校准点")
    print("=" * 108)
    print("  起始:", smi())
    if "--case" in sys.argv:
        print(run_one(int(sys.argv[sys.argv.index("--case") + 1])))
        return
    import subprocess
    for i in range(len(CASES)):
        r = subprocess.run([sys.executable, "-u", __file__, "--case", str(i)],
                           capture_output=True, text=True, timeout=300)
        line = [l for l in r.stdout.strip().split("\n") if l.startswith("RESULT")]
        print("  " + (line[0] if line else "CASE %d FAILED: %s" % (i, r.stderr.strip()[-120:])))
    print("  结束:", smi())


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
