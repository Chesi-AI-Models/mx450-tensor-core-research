# -*- coding: utf-8 -*-
"""D1 续: float2/float4 向量化加载 —— 从 61.3% 继续往上

联网查证的量化参考（N=4096, HGEMM 渐进优化）:
  baseline 标量加载     42 TFLOPS (15% cuBLAS)
  + float4 向量化       74 TFLOPS (27% cuBLAS)  <- 1.8x
  + cp.async            82 TFLOPS (30%)
  + 共享内存 padding    ...

★ 关键几何约束（必须自己算，不能照抄）:
  float4 = 16 字节 = 8 个 half，要求地址 16 字节对齐
  A tile 用 PAD=6 -> 行跨度 (BK+6) half
    BK=8  -> 14 half = 28 bytes，28 % 16 != 0  -> **float4 会踩未定义行为**
  但 28 % 4 == 0  -> **float2（4字节/2个half）安全**
  且 bank 分析: ty 差 8 行 -> 8*14*2/4 = 56 banks, 56 % 32 = 24 -> 不冲突 ✓
  B tile 无 padding，行跨度 BN=128 half = 256 bytes = 16 的倍数 -> **可用 float4**

本脚本对比:
  V0  基线（标量加载, BK=8 PAD=6）        <- 上轮最优 1.876 TFLOPS
  V1  A:float2 + B:float2
  V2  A:float2 + B:float4   <- 预期最优
  V3  A:float4 + B:float4   <- 预期对齐失败（验证假设）
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

    def __getattr__(self, name):
        return getattr(self.cu, name)

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
            raise RuntimeError("cuMemAlloc rc=%d" % rc)
        return p


HEAD = r"""
__device__ __forceinline__ float h2f(unsigned short h) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
#define BM 128
#define BN 128
#define PAD __PAD__
#define TM 8
#define TN 8
#define NTHREADS 256
"""

TAIL = r"""
    for (int k0 = 0; k0 < K; k0 += __BK__) {
__LOAD_A__
__LOAD_B__
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < __BK__; kk++) {
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

def align_ok(BK, PAD, a_mode, b_mode, BM=128, BN=128):
    """静态对齐预检 —— 不能靠跑崩来验证（misaligned 是粘性错误，会打死 context）"""
    def need(m):
        return {"scalar": 1, "v2": 4, "v4": 16}[m]
    msgs = []
    for tag, mode, stride, rows in (("A", a_mode, BK + PAD, BM),
                                    ("B", b_mode, BN, BK)):
        al = need(mode)
        vec = {"scalar": 1, "v2": 2, "v4": 8}[mode]
        for r in (0, 1, 2, 3, rows - 1):
            for c in range(0, stride, vec):
                if (r * stride + c) * 2 % al:
                    msgs.append("%s共享内存 r=%d 需 %dB 对齐但偏移 %dB"
                                % (tag, r, al, (r * stride + c) * 2 % al))
                    break
            else:
                continue
            break
    return msgs


def make_kernel(BK, PAD, a_mode, b_mode):
    """a_mode / b_mode: 'scalar' | 'v2' | 'v4'"""
    decl = r"""
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
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;
"""
    decl = decl.replace("BK + PAD", "__BK__ + PAD").replace("[BK]", "[__BK__]")

    # --- A 加载
    if a_mode == "scalar":
        la = r"""
        #pragma unroll
        for (int t = 0; t < (BM * __BK__) / NTHREADS; t++) {
            int idx = threadIdx.x + t * NTHREADS;
            int r = idx / __BK__, c = idx % __BK__;
            int gr = row0 + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0;
        }"""
    elif a_mode == "v2":
        # 2 个 half = 4 bytes。注意 float2 是 2 个 float = 8 bytes，不能用！
        la = r"""
        {
            constexpr int AV = (BM * __BK__) / 2;      // u32 向量总数
            #pragma unroll
            for (int t = 0; t < (AV + NTHREADS - 1) / NTHREADS; t++) {
                int v = threadIdx.x + t * NTHREADS;
                if (v >= AV) break;
                int r = v / (__BK__ / 2);
                int c2 = (v % (__BK__ / 2)) * 2;
                int gr = row0 + r;
                int gc = k0 + c2;
                unsigned int val;
                if (gr < M && gc + 1 < K) {
                    val = *reinterpret_cast<const unsigned int*>(&A[gr * K + gc]);
                } else {
                    val = 0u;
                }
                *reinterpret_cast<unsigned int*>(&As[r][c2]) = val;
            }
        }"""
    else:  # v4
        la = r"""
        {
            constexpr int AV = (BM * __BK__) / 8;      // float4 总数
            #pragma unroll
            for (int t = 0; t < (AV + NTHREADS - 1) / NTHREADS; t++) {
                int v = threadIdx.x + t * NTHREADS;
                if (v >= AV) break;
                int r = v / (__BK__ / 8);
                int c8 = (v % (__BK__ / 8)) * 8;
                int gr = row0 + r;
                int gc = k0 + c8;
                if (gr < M && gc + 7 < K) {
                    *reinterpret_cast<float4*>(&As[r][c8]) =
                        *reinterpret_cast<const float4*>(&A[gr * K + gc]);
                } else {
                    for (int q = 0; q < 8; q++)
                        As[r][c8 + q] = (gr < M && gc + q < K) ? A[gr * K + gc + q] : 0;
                }
            }
        }"""

    # --- B 加载
    if b_mode == "scalar":
        lb = r"""
        #pragma unroll
        for (int t = 0; t < (__BK__ * BN) / NTHREADS; t++) {
            int idx = threadIdx.x + t * NTHREADS;
            int r = idx / BN, c = idx % BN;
            int gr = k0 + r, gc = col0 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0;
        }"""
    elif b_mode == "v2":
        lb = r"""
        {
            constexpr int BV = (__BK__ * BN) / 2;
            #pragma unroll
            for (int t = 0; t < (BV + NTHREADS - 1) / NTHREADS; t++) {
                int v = threadIdx.x + t * NTHREADS;
                if (v >= BV) break;
                int r = v / (BN / 2);
                int c2 = (v % (BN / 2)) * 2;
                int gr = k0 + r;
                int gc = col0 + c2;
                unsigned int val;
                if (gr < K && gc + 1 < N) {
                    val = *reinterpret_cast<const unsigned int*>(&B[gr * N + gc]);
                } else {
                    val = 0u;
                }
                *reinterpret_cast<unsigned int*>(&Bs[r][c2]) = val;
            }
        }"""
    else:  # v4
        lb = r"""
        {
            constexpr int BV = (__BK__ * BN) / 8;
            #pragma unroll
            for (int t = 0; t < (BV + NTHREADS - 1) / NTHREADS; t++) {
                int v = threadIdx.x + t * NTHREADS;
                if (v >= BV) break;
                int r = v / (BN / 8);
                int c8 = (v % (BN / 8)) * 8;
                int gr = k0 + r;
                int gc = col0 + c8;
                if (gr < K && gc + 7 < N) {
                    *reinterpret_cast<float4*>(&Bs[r][c8]) =
                        *reinterpret_cast<const float4*>(&B[gr * N + gc]);
                } else {
                    for (int q = 0; q < 8; q++)
                        Bs[r][c8 + q] = (gr < K && gc + q < N) ? B[gr * N + gc + q] : 0;
                }
            }
        }"""

    body = (HEAD.replace("__PAD__", str(PAD))
            + decl.replace("__BK__", str(BK))
            + TAIL.replace("__LOAD_A__", la).replace("__LOAD_B__", lb)
            .replace("__BK__", str(BK)))
    return body.replace("__BK__", str(BK)).replace("__PAD__", str(PAD))


def main():
    print("=" * 78)
    print("D1 续: 向量化加载（float2 / float4）")
    print("=" * 78)
    torch.cuda.init()
    p = torch.cuda.get_device_properties(0)
    print("  GPU:", p.name, "CC=%d.%d" % (p.major, p.minor))

    N = 2048
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16)
    ref = torch.mm(a.float(), b.float())

    cu = Cuda()
    nv = Nvrtc()
    a_p, b_p = a.data_ptr(), b.data_ptr()
    c_dev = cu.malloc(N * N * 2)

    print()
    print("  %-34s %10s %10s %8s %10s"
          % ("配置", "耗时(ms)", "TFLOPS", "效率", "max_err"))
    print("  " + "-" * 78)

    results = []
    cases = [
        ("V0 基线 标量 BK=8 PAD=6", 8, 6, "scalar", "scalar"),
        ("V1 A:v2  B:v2  BK=8 PAD=6", 8, 6, "v2", "v2"),
        ("V2 A:v2  B:v4  BK=8 PAD=6", 8, 6, "v2", "v4"),
        ("V2b A:v2 B:v4  BK=16 PAD=6", 16, 6, "v2", "v4"),
        ("V3 A:v4  B:v4  BK=8 PAD=6", 8, 6, "v4", "v4"),
        ("V4 A:v4  B:v4  BK=8 PAD=8", 8, 8, "v4", "v4"),
        ("V5 A:v2  B:v4  BK=8 PAD=10", 8, 10, "v2", "v4"),
        ("V6 A:v4  B:v4  BK=16 PAD=8", 16, 8, "v4", "v4"),
    ]

    for label, BK, PAD, am, bm in cases:
        bad = align_ok(BK, PAD, am, bm)
        if bad:
            print("  %-34s SKIP(对齐预检): %s" % (label, bad[0]))
            continue
        src = make_kernel(BK, PAD, am, bm)
        try:
            ptx = nv.compile(src)
        except Exception as e:
            print("  %-34s COMPILE FAIL: %s" % (label, str(e)[:50].replace("\n", " ")))
            continue
        try:
            mod, fn = cu.load(ptx)
        except Exception as e:
            print("  %-34s LOAD FAIL: %s" % (label, str(e)[:50]))
            continue

        grid = ((N + 127) // 128, (N + 127) // 128, 1)
        kp = [ctypes.c_uint64(a_p), ctypes.c_uint64(b_p), ctypes.c_uint64(c_dev.value),
              ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(N)]
        pp = (ctypes.c_void_p * len(kp))(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in kp])

        def launch():
            cu.cuLaunchKernel(fn, grid[0], grid[1], grid[2], 256, 1, 1, 0, None, pp, None)

        try:
            launch()
            cu.cuCtxSynchronize()
            out = torch.empty(N, N, device="cuda", dtype=torch.float16)
            cu.cuMemcpyDtoH_v2(ctypes.c_void_p(out.data_ptr()), c_dev.value, N * N * 2)
            err = (out.float() - ref).abs().max().item()
        except Exception as e:
            print("  %-34s RUN FAIL: %s" % (label, str(e)[:50]))
            cu.cuModuleUnload(mod)
            continue
        if err > 1.0:
            print("  %-34s WRONG err=%.3f（对齐假设可能失败）" % (label, err))
            cu.cuModuleUnload(mod)
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
        print("  %-34s %10.3f %10.4f %7.1f%% %10.5f" % (label, dt * 1e3, tf, eff, err))
        results.append((label, tf, eff))
        cu.cuModuleUnload(mod)

    cu.cuMemFree_v2(c_dev)

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    if results:
        base = results[0]
        best = max(results, key=lambda x: x[1])
        print("  基线 %-30s %.4f TFLOPS (%.1f%%)" % (base[0], base[1], base[2]))
        print("  最优 %-30s %.4f TFLOPS (%.1f%%)  -> %.2fx"
              % (best[0], best[1], best[2], best[1] / base[1]))
        print("  目标 2.5 TFLOPS (85%%)  达成 %.0f%%" % (best[1] / 2.5 * 100))
        print()
        print("  历史: 1.603 (52.3%%) -> 1.876 (61.3%%) -> 本次 %.4f (%.1f%%)"
              % (best[1], best[2]))
        print("  参考 cuBLAS fp32 甜区 2.37")
        if best[1] > 2.37:
            print("  >>> 已超过 cuBLAS fp32")
        else:
            print("  >>> 仍低于 cuBLAS fp32（差 %.0f%%）" % ((1 - best[1] / 2.37) * 100))


if __name__ == "__main__":
    main()
