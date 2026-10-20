# -*- coding: utf-8 -*-
"""D2 超频/锁频实验 —— 唯一能超过「3.06 TFLOPS」的路径

前提事实（实测）:
  clocks.max.sm      = 2100 MHz
  clocks.sm 空闲      = 1395 MHz
  负载下实际         = 1260-1395 MHz（掉频）
  纯 FP32 FFMA 达到时钟折算峰值的 99.0% -> 公式 2*64*14*clk 可信
  -> 若能把时钟锁到 1830/2100，峰值 = 3.28 / 3.76 TFLOPS

本脚本:
  1. 打印 SUPPORTED_CLOCKS / 当前锁频状态 / 持久模式
  2. 依次尝试 -lgc <clk>（锁定 SM 时钟），每档跑一次 GEMM 与纯 FMA
  3. 全部结束后 -rgc 还原（无论成功失败）
用法: sudo python3 d2_clock_lock.py
"""
import ctypes, glob, statistics, subprocess, threading, time
import torch


def sh(cmd, timeout=25):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.returncode, (r.stdout + r.stderr).strip())
    except Exception as e:
        return (99, str(e))


def _find(pat):
    for p in pat:
        h = glob.glob(p)
        if h:
            return h[0]
    return None


def sample(stop, out):
    while not stop.is_set():
        try:
            v = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=3).stdout.strip()
            p = [x.strip() for x in v.split(",")]
            out.append((int(p[0]), p[1]))
        except Exception:
            pass
        time.sleep(0.08)


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

    def __getattr__(self, n):
        return getattr(self.cu, n)

    def load(self, ptx, name):
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
    for (int t = 0; t < 20000; t++) {
        #pragma unroll
        for (int i = 0; i < NACC; i++) a[i] = fmaf(a[i], x, y);
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < NACC; i++) s += a[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
"""


def run(launch, flop, secs=3.0):
    launch()
    torch.cuda.synchronize()
    clk = []
    stop = threading.Event()
    th = threading.Thread(target=sample, args=(stop, clk))
    th.start()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs:
        launch()
        n += 1
    torch.cuda.synchronize()
    stop.set()
    th.join()
    dt = time.perf_counter() - t0
    cl = [c[0] for c in clk] or [0]
    med = statistics.median(cl)
    tf = flop * n / dt / 1e12
    peak = 2 * 64 * 14 * med * 1e6 / 1e12
    return tf, med, min(cl), max(cl), (tf / peak * 100 if peak else 0)


def main():
    print("=" * 100)
    print("D2 锁频 / 超频实验")
    print("=" * 100)
    print("  当前用户:", sh("id -u")[1])
    print("  --- 支持的时钟档位（前 12 行）---")
    rc, txt = sh("nvidia-smi -q -d SUPPORTED_CLOCKS | head -20")
    print("\n".join("    " + l for l in txt.split("\n")[:20]))
    print("  --- 当前持久模式/应用时钟 ---")
    print("   ", sh("nvidia-smi -q -d PERFORMANCE | grep -E 'Persistence|Clocks (Event|Throttle)' -A2")[1][:400])

    N = 2048
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16)
    cu, nv = Cuda(), Nvrtc()
    c_dev = cu.malloc(N * N * 2)
    mod, fn = cu.load(nv.compile(GEMM), b"gemm_ffma")
    grid = ((N + 127) // 128, (N + 63) // 64, 1)
    kp = [ctypes.c_uint64(a.data_ptr()), ctypes.c_uint64(b.data_ptr()),
          ctypes.c_uint64(c_dev.value), ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(N)]
    pp = (ctypes.c_void_p * 6)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in kp])

    def g():
        cu.cuLaunchKernel(fn, grid[0], grid[1], grid[2], 128, 1, 1, 0, None, pp, None)

    mod2, fn2 = cu.load(nv.compile(FMA), b"fma_peak")
    inb = torch.randn(64, device="cuda", dtype=torch.float32).abs() + 1.0
    ob = torch.zeros(14 * 4 * 256, device="cuda", dtype=torch.float32)
    kp2 = [ctypes.c_uint64(inb.data_ptr()), ctypes.c_uint64(ob.data_ptr()),
           ctypes.c_int(0), ctypes.c_int(0)]
    pp2 = (ctypes.c_void_p * 4)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in kp2])
    fl = 2.0 * 32 * 20000 * 256 * 14 * 4

    def f():
        cu.cuLaunchKernel(fn2, 14 * 4, 1, 1, 256, 1, 1, 0, None, pp2, None)

    print()
    print("  %-22s %9s %9s %8s %9s" % ("阶段", "GEMM TF", "GEMM时钟", "GEMM%", "FMA TF"))
    print("  " + "-" * 70)

    rows = []

    def stage(label):
        tg, mg, mng, mxg, pg = run(g, 2.0 * N ** 3)
        tf_, mf, mnf, mxf, pf = run(f, fl, secs=2.0)
        print("  %-22s %9.4f %7d MHz %7.1f%% %9.4f" % (label, tg, mg, pg, tf_))
        rows.append((label, tg, mg, pg, tf_, mf, pf))
        return tg

    stage("默认（未锁频）")

    for clk in (1500, 1650, 1800, 1950, 2100):
        rc, out = sh("nvidia-smi -lgc %d" % clk)
        ok = (rc == 0)
        print("  [lock %d MHz] rc=%d %s" % (clk, rc, out.replace("\n", " ")[:80]))
        if not ok:
            continue
        time.sleep(1.0)
        stage("锁定 %d MHz" % clk)

    rc, out = sh("nvidia-smi -rgc")
    print("  [解锁] rc=%d %s" % (rc, out.replace("\n", " ")[:80]))
    time.sleep(1.0)

    print()
    print("  最终时钟:", sh("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu "
                            "--format=csv,noheader")[1])
    print()
    print("  %-22s %9s %9s %8s %9s" % ("汇总", "GEMM TF", "时钟", "效率", "FMA TF"))
    print("  " + "-" * 70)
    for r in rows:
        print("  %-22s %9.4f %7d MHz %7.1f%% %9.4f" % (r[0], r[1], r[2], r[3], r[4]))
    best = max(rows, key=lambda x: x[1]) if rows else None
    if best:
        print()
        print("  最优 %s: GEMM %.4f TFLOPS @ %d MHz" % (best[0], best[1], best[2]))
        print("  历史最好（未锁频历史记录 2.4872）-> %+.1f%%"
              % ((best[1] / 2.4872 - 1) * 100))
        print("  「3.06 TFLOPS」线: %s" % ("已超过" if best[1] > 3.06 else "未超过"))


if __name__ == "__main__":
    main()
