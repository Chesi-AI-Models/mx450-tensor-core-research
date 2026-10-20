# -*- coding: utf-8 -*-
"""D1 决定性测量: 抓「内核运行期间」的真实 SM 时钟

矛盾:
  nvidia-smi 空闲报 1395 MHz，但测纯 FFMA 时测后报 1830 MHz，max 是 2100 MHz。
  纯 FFMA 微基准 2.4170 TFLOPS < GEMM 实测 2.4872 TFLOPS  -> 分母在漂，不可能。

做法: 前台线程持续发射 kernel，后台线程每 100ms 采一次 clocks.sm，
      得到「运行期间」的时钟分布(min/中位/max) -> 反算真实效率。
"""
import ctypes, glob, statistics, subprocess, threading, time
import torch


def _find(pat):
    for p in pat:
        h = glob.glob(p)
        if h:
            return h[0]
    return None


def sample_clock(stop, out):
    while not stop.is_set():
        try:
            v = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=3).stdout.strip()
            parts = [x.strip() for x in v.split(",")]
            out.append((int(parts[0]), parts[1], parts[2]))
        except Exception:
            pass
        time.sleep(0.1)


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
        cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        cu.cuCtxSynchronize.argtypes = []
        cu.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                          ctypes.c_void_p]

    def __getattr__(self, n):
        return getattr(self.cu, n)

    def load(self, ptx, name=b"gemm_ffma"):
        mod = ctypes.c_void_p()
        if self.cu.cuModuleLoadData(ctypes.byref(mod), ptx.encode()):
            raise RuntimeError("load")
        fn = ctypes.c_void_p()
        if self.cu.cuModuleGetFunction(ctypes.byref(fn), mod, name):
            raise RuntimeError("getfn")
        return mod, fn

    def malloc(self, n):
        p = ctypes.c_uint64()
        if self.cu.cuMemAlloc_v2(ctypes.byref(p), n):
            raise RuntimeError("malloc")
        return p


GEMM = r"""
#define BM 64
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define PAD 0
#define NT 128
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
        int r = v / (BK / 8), c8 = (v % (BK / 8)) * 8;
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
        int r = v / (BN / 8), c8 = (v % (BN / 8)) * 8;
        int gr = k0 + r, gc = col0 + c8;
        if (gr < K && gc + 7 < N)
            *reinterpret_cast<uint4*>(&Bs[r][c8]) = *reinterpret_cast<const uint4*>(&B[gr * N + gc]);
        else
            for (int q = 0; q < 8; q++)
                Bs[r][c8 + q] = (gr < K && gc + q < N) ? B[gr * N + gc + q] : 0;
    }
}
extern "C" __global__ void gemm_ffma(const unsigned short* __restrict__ A,
                                     const unsigned short* __restrict__ B,
                                     unsigned short* __restrict__ C, int M, int N, int K)
{
    __shared__ unsigned short As[BM][BK + PAD];
    __shared__ unsigned short Bs[BK][BN];
    const int tx = threadIdx.x % NX, ty = threadIdx.x / NX;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tile(A, B, As, Bs, row0, col0, k0, M, N, K);
        __syncthreads();
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
        __syncthreads();
    }
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

FMA = r"""
#define NACC 32
extern "C" __global__ void fma_peak(const float* __restrict__ in, float* __restrict__ out,
                                    int n, int dummy)
{
    float x = in[threadIdx.x & 31] + 1.0000001f;
    float y = in[(threadIdx.x + 7) & 31] * 1e-7f;
    float a[NACC];
    #pragma unroll
    for (int i = 0; i < NACC; i++) a[i] = in[(threadIdx.x + i) & 31];
    #pragma unroll 1
    for (int t = 0; t < 30000; t++) {
        #pragma unroll
        for (int i = 0; i < NACC; i++) a[i] = fmaf(a[i], x, y);
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < NACC; i++) s += a[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
"""


def measure(name, launch, flop_per_launch, seconds=4.0):
    launch()
    torch.cuda.synchronize()
    clk = []
    stop = threading.Event()
    th = threading.Thread(target=sample_clock, args=(stop, clk))
    th.start()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        launch()
        n += 1
    torch.cuda.synchronize()
    stop.set()
    th.join()
    dt = time.perf_counter() - t0
    cl = [c[0] for c in clk] or [0]
    tf = flop_per_launch * n / dt / 1e12
    med = statistics.median(cl)
    return dict(name=name, tf=tf, n=n, dt=dt, med=med, mn=min(cl), mx=max(cl),
                temp=sorted(set(c[1] for c in clk))[:3], pw=sorted(set(c[2] for c in clk))[:3])


def main():
    print("=" * 96)
    print("D1 决定性测量: 内核运行期间的真实时钟")
    print("=" * 96)
    N = 2048
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16)
    af, bf = a.float(), b.float()
    cu, nv = Cuda(), Nvrtc()
    c_dev = cu.malloc(N * N * 2)

    out = []

    # 1. 自研 GEMM (C7 配置)
    mod, fn = cu.load(nv.compile(GEMM))
    grid = ((N + 127) // 128, (N + 63) // 64, 1)
    kp = [ctypes.c_uint64(a.data_ptr()), ctypes.c_uint64(b.data_ptr()),
          ctypes.c_uint64(c_dev.value), ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(N)]
    pp = (ctypes.c_void_p * 6)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in kp])
    out.append(measure("自研 GEMM 64x128x8",
                       lambda: cu.cuLaunchKernel(fn, grid[0], grid[1], grid[2], 128, 1, 1,
                                                 0, None, pp, None),
                       2.0 * N ** 3))

    # 2. cuBLAS fp32
    out.append(measure("cuBLAS fp32 mm", lambda: torch.mm(af, bf), 2.0 * N ** 3))

    # 3. cuBLAS fp16
    out.append(measure("cuBLAS fp16 mm", lambda: torch.mm(a, b), 2.0 * N ** 3))

    # 4. 纯 FMA
    mod2, fn2 = cu.load(nv.compile(FMA), b"fma_peak")
    inb = torch.randn(64, device="cuda", dtype=torch.float32).abs() + 1.0
    ob = torch.zeros(14 * 4 * 256, device="cuda", dtype=torch.float32)
    kp2 = [ctypes.c_uint64(inb.data_ptr()), ctypes.c_uint64(ob.data_ptr()),
           ctypes.c_int(0), ctypes.c_int(0)]
    pp2 = (ctypes.c_void_p * 4)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in kp2])
    fl = 2.0 * 32 * 30000 * 256 * 14 * 4
    out.append(measure("纯 FP32 FFMA",
                       lambda: cu.cuLaunchKernel(fn2, 14 * 4, 1, 1, 256, 1, 1, 0, None, pp2, None),
                       fl, seconds=3.0))

    print()
    print("  %-20s %9s %10s %10s %8s %6s"
          % ("项目", "TFLOPS", "中位时钟", "最低-最高", "温度", "功耗"))
    print("  " + "-" * 76)
    for r in out:
        print("  %-20s %9.4f %7d MHz %5d-%-5d %8s %6s"
              % (r["name"], r["tf"], r["med"], r["mn"], r["mx"],
                 r["temp"][0] if r["temp"] else "?", r["pw"][0] if r["pw"] else "?"))

    print()
    print("  用「运行期间中位时钟」反算真实效率（2*64*14*clk 作分母）:")
    for r in out:
        peak = 2 * 64 * 14 * r["med"] * 1e6 / 1e12
        print("    %-20s %.4f / %.4f = %5.1f%%"
              % (r["name"], r["tf"], peak, r["tf"] / peak * 100 if peak else 0))
    print()
    print("  参考: 时钟-峰值对照表（2*64*14*clk）")
    for c in (1290, 1395, 1560, 1665, 1785, 1830, 1950, 2100):
        print("    %4d MHz -> %.3f TFLOPS" % (c, 2 * 64 * 14 * c * 1e6 / 1e12))
    print()
    print("  结论口径:")
    print("    * 若自研 GEMM 效率 >= 95%，则 D1 已到该时钟下的硬件上限，")
    print("      继续优化需要提高时钟 -> 转入 D2 超频（max_SM 报 2100 MHz，当前仅 1395-1830）")
    print("    * 「3.06 TFLOPS / 85%」这个目标基于 1665 MHz 假设，与实测时钟不符，须废弃")


if __name__ == "__main__":
    main()
