# -*- coding: utf-8 -*-
"""D1 第三步: 共享内存存 float + A tile 转置 -> 内层 LDS.128

★ 联网查证的机制（ppc.cs.aalto.fi/ch4/v4 与 baizeway/swizzle 文章）:
   - float4 只有 16B 对齐才合法，未对齐 = 崩溃或读脏（CUDA 没有非对齐向量 load）
   - padding 会破坏 16B 对齐 -> 「padding + float4 写 smem」不可行，必须拆成标量 STS
   - 真正的收益来自 **内层循环的 smem 读取向量化（LDS.128）**，
     做法是 **把 A tile 转置存 smem**，让 a[i] 沿 K 连续

★ 上一轮实测（v5）:
   V6 A:v4 B:v4 BK=16 PAD=8 = 1.9714 TFLOPS (64.4%)  <- 只改了 global->smem 的带宽
   内层仍是 16x LDS.U16 + 16x cvt.f32.f16 + 64x FFMA  -> 非 FFMA 占比过高

★ 本轮把内层改成:
   W2/W3/W4:  smem 存 float，A 转置为 [BK][BM+PADT]
              a[] 用 2x float4 读、b[] 用 2x float4 读
              -> 每 kk: 4x LDS.128 + 64x FFMA，非 FFMA 占比 5.9%
   W1:        对照组，smem 存 float 但 A 不转置（a[i] 仍是列跨步，只能 LDS.32）
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


# ---------------------------------------------------------------- 代码模板
PRELUDE = r"""
__device__ __forceinline__ float h2f(unsigned short h) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
// 一个 uint4 = 16B = 8 个 half -> 8 个 float
__device__ __forceinline__ void unpack8(uint4 v, float* o) {
    unsigned int w[4]; w[0]=v.x; w[1]=v.y; w[2]=v.z; w[3]=v.w;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        o[2*i+0] = h2f((unsigned short)(w[i] & 0xffffu));
        o[2*i+1] = h2f((unsigned short)(w[i] >> 16));
    }
}
#define BM 128
#define BN 128
#define TM 8
#define TN 8
#define NTHREADS 256
"""

DECL = r"""
extern "C" __global__ void gemm_ffma(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ B,
    unsigned short* __restrict__ C,
    int M, int N, int K)
{
__SMDECL__
    const int tx = threadIdx.x % 16;
    const int ty = threadIdx.x / 16;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;
__TAILBODY__
"""

_load_global = r"""
        #pragma unroll
        for (int t = 0; t < (BM * __BK__ / 8 + NTHREADS - 1) / NTHREADS; t++) {
            int v = threadIdx.x + t * NTHREADS;
            if (v >= BM * __BK__ / 8) break;
            int r = v / (__BK__ / 8);
            int c8 = (v % (__BK__ / 8)) * 8;
            int gr = row0 + r, gc = k0 + c8;
            float f[8];
            if (gr < M && gc + 7 < K) {
                unpack8(*reinterpret_cast<const uint4*>(&A[gr * K + gc]), f);
            } else {
                #pragma unroll
                for (int q = 0; q < 8; q++)
                    f[q] = (gr < M && gc + q < K) ? h2f(A[gr * K + gc + q]) : 0.f;
            }
__STORE_A__
        }
        #pragma unroll
        for (int t = 0; t < (__BK__ * BN / 8 + NTHREADS - 1) / NTHREADS; t++) {
            int v = threadIdx.x + t * NTHREADS;
            if (v >= __BK__ * BN / 8) break;
            int r = v / (BN / 8);
            int c8 = (v % (BN / 8)) * 8;
            int gr = k0 + r, gc = col0 + c8;
            float f[8];
            if (gr < K && gc + 7 < N) {
                unpack8(*reinterpret_cast<const uint4*>(&B[gr * N + gc]), f);
            } else {
                #pragma unroll
                for (int q = 0; q < 8; q++)
                    f[q] = (gr < K && gc + q < N) ? h2f(B[gr * N + gc + q]) : 0.f;
            }
            *reinterpret_cast<float4*>(&Bs[r][c8])     = make_float4(f[0], f[1], f[2], f[3]);
            *reinterpret_cast<float4*>(&Bs[r][c8 + 4]) = make_float4(f[4], f[5], f[6], f[7]);
        }
"""

CORE_VEC = r"""
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < __BK__; kk++) {
            float a[TM], b[TN];
            {
                const float4* pa = reinterpret_cast<const float4*>(&AsT[kk][ty * TM]);
                float4 a0 = pa[0], a1 = pa[1];
                a[0]=a0.x; a[1]=a0.y; a[2]=a0.z; a[3]=a0.w;
                a[4]=a1.x; a[5]=a1.y; a[6]=a1.z; a[7]=a1.w;
            }
            {
                const float4* pb = reinterpret_cast<const float4*>(&Bs[kk][tx * TN]);
                float4 b0 = pb[0], b1 = pb[1];
                b[0]=b0.x; b[1]=b0.y; b[2]=b0.z; b[3]=b0.w;
                b[4]=b1.x; b[5]=b1.y; b[6]=b1.z; b[7]=b1.w;
            }
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
"""

CORE_SCALAR = r"""
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < __BK__; kk++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty * TM + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
"""

EPI = r"""
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


def make_kernel(BK, PADT, transpose, vec_read):
    """transpose=True -> A 存成 [BK][BM+PADT] 的 float（转置）
       transpose=False -> A 存成 [BM][BK+PADT] 的 float（不转置，对照）
       vec_read=True -> 内层用 float4 读（只有 transpose 时 a[] 才连续）"""
    if transpose:
        smdecl = "    __shared__ float AsT[__BK__][BM + %d];\n" % PADT
        store_a = ("            #pragma unroll\n"
                   "            for (int q = 0; q < 8; q++) AsT[c8 + q][r] = f[q];")
    else:
        smdecl = "    __shared__ float As[BM][__BK__ + %d];\n" % PADT
        store_a = ("            *reinterpret_cast<float4*>(&As[r][c8])     = "
                   "make_float4(f[0], f[1], f[2], f[3]);\n"
                   "            *reinterpret_cast<float4*>(&As[r][c8 + 4]) = "
                   "make_float4(f[4], f[5], f[6], f[7]);")
    smdecl += "    __shared__ float Bs[__BK__][BN];"

    body = ("        for (int k0 = 0; k0 < K; k0 += __BK__) {\n"
            + _load_global.replace("__STORE_A__", store_a)
            + (CORE_VEC if vec_read else CORE_SCALAR)
            + "        }\n"
            + EPI)
    src = PRELUDE + DECL.replace("__SMDECL__", smdecl).replace("__TAILBODY__", body)
    return src.replace("__BK__", str(BK))


def check_align(BK, PADT, transpose, vec_read):
    """静态对齐预检：float 32 位，float4 需 16B"""
    if not vec_read:
        return []
    if transpose:
        stride = 128 + PADT
        if stride % 4:
            return ["AsT 行跨度 %d 不是 4 的倍数 -> float4 无法对齐" % stride]
        # &AsT[kk][ty*8] -> (kk*stride + ty*8)*4 需 %16 == 0
        for kk in range(BK):
            if (kk * stride) * 4 % 16:
                return ["AsT kk=%d 偏移 %dB 非 16 倍数" % (kk, kk * stride * 4)]
        return []
    else:
        stride = BK + PADT
        if stride % 4:
            return ["As 行跨度 %d 不是 4 的倍数 -> float4 无法对齐" % stride]
        return []


def main():
    print("=" * 84)
    print("D1 第三步: smem 存 float + A 转置 -> 内层 LDS.128")
    print("=" * 84)
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
    print("  %-40s %9s %9s %7s %10s"
          % ("配置", "耗时(ms)", "TFLOPS", "效率", "max_err"))
    print("  " + "-" * 84)

    cases = [
        ("W0 对照 半精度smem BK=16 PAD=8", 16, 8, False, False),
        ("W1 float smem 不转置 BK=16 PAD=4", 16, 4, False, False),
        ("W2 float smem 转置 BK=16 PADT=4", 16, 4, True, True),
        ("W3 float smem 转置 BK=8  PADT=4", 8, 4, True, True),
        ("W4 float smem 转置 BK=16 PADT=12", 16, 12, True, True),
        ("W5 float smem 转置 BK=32 PADT=4", 32, 4, True, True),
    ]

    results = []
    for label, BK, PADT, tr, vr in cases:
        bad = check_align(BK, PADT, tr, vr)
        if bad:
            print("  %-40s SKIP: %s" % (label, bad[0]))
            continue
        try:
            src = make_kernel(BK, PADT, tr, vr)
            ptx = nv.compile(src)
        except Exception as e:
            msg = str(e).replace("\n", " ")
            print("  %-40s COMPILE FAIL: %s" % (label, msg[:60]))
            continue
        try:
            mod, fn = cu.load(ptx)
        except Exception as e:
            print("  %-40s LOAD FAIL: %s" % (label, str(e)[:50]))
            continue

        grid = ((N + 127) // 128, (N + 127) // 128, 1)
        kp = [ctypes.c_uint64(a_p), ctypes.c_uint64(b_p), ctypes.c_uint64(c_dev.value),
              ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(N)]
        pp = (ctypes.c_void_p * len(kp))(
            *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in kp])

        def launch():
            cu.cuLaunchKernel(fn, grid[0], grid[1], grid[2], 256, 1, 1, 0, None, pp, None)

        try:
            launch()
            cu.cuCtxSynchronize()
            out = torch.empty(N, N, device="cuda", dtype=torch.float16)
            cu.cuMemcpyDtoH_v2(ctypes.c_void_p(out.data_ptr()), c_dev.value, N * N * 2)
            err = (out.float() - ref).abs().max().item()
        except Exception as e:
            print("  %-40s RUN FAIL: %s" % (label, str(e)[:60]))
            cu.cuModuleUnload(mod)
            break   # 粘性错误，后面全废，直接退出循环
        if err > 1.0:
            print("  %-40s WRONG err=%.3f" % (label, err))
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
        tf = 2.0 * N ** 3 / dt / 1e12
        eff = tf / 3.06 * 100
        print("  %-40s %9.3f %9.4f %6.1f%% %10.5f" % (label, dt * 1e3, tf, eff, err))
        results.append((label, tf, eff))
        cu.cuModuleUnload(mod)

    cu.cuMemFree_v2(c_dev)

    print()
    print("=" * 84)
    print("结论")
    print("=" * 84)
    if results:
        base = results[0]
        best = max(results, key=lambda x: x[1])
        print("  对照 %-32s %.4f TFLOPS (%.1f%%)" % (base[0], base[1], base[2]))
        print("  最优 %-32s %.4f TFLOPS (%.1f%%)  -> 相对对照 %.2fx"
              % (best[0], best[1], best[2], best[1] / base[1]))
        print("  目标 2.5 TFLOPS (85%) 达成 %.0f%%" % (best[1] / 2.5 * 100))
        print()
        print("  历史: 1.603(52.3%) -> 1.876(61.3%) -> 1.971(64.4%) -> 本次 %.4f(%.1f%%)"
              % (best[1], best[2]))
        print("  参考 cuBLAS fp32 甜区 2.37")
        if best[1] > 2.37:
            print("  >>> 已超过 cuBLAS fp32")
        else:
            print("  >>> 仍低于 cuBLAS fp32（差 %.0f%%）" % ((1 - best[1] / 2.37) * 100))


if __name__ == "__main__":
    main()
