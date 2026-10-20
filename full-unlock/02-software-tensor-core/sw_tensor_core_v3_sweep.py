# -*- coding: utf-8 -*-
"""D1: 软件 Tensor Core GEMM 优化 —— 从 1.603 TFLOPS 榨向 2.5

现状（sw_tensor_core_v2.py）: BM=BN=128, BK=16, TM=TN=8, 256 线程 -> 1.603 TFLOPS = 52% of 3.06

已识别的三个瓶颈:
  1. As[128][16] bank conflict
     行跨度 S=16 short = 32 bytes = 8 banks
     ty 相邻 -> 8 行 -> 8*16*2/4 = 64 banks ≡ 0 (mod 32) -> 同 warp 撞同一 bank
     修复条件: 4*S % 32 != 0  即 S % 8 != 0
  2. 计算访存比仅 4:1 (每 kk 做 64 FMA, 读 16 个值)
  3. BK=16 太短 -> __syncthreads 次数 = K/16 = 128 次 (K=2048)

本脚本扫描 4 个配置:
  (BK=16, PAD=0)  = v2 基线
  (BK=32, PAD=0)  减少 sync 次数
  (BK=32, PAD=4)  同时消 bank conflict
  (BK=64, PAD=4)  进一步减少 sync
"""
import ctypes, glob, time, sys
import torch

# ============================================================ NVRTC
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
        opts = [("--gpu-architecture=" + arch).encode(), b"--std=c++14",
                b"--use_fast_math"]
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

# ============================================================ CUDA
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
        cu.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
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

# ============================================================ kernel 模板
KERNEL = r"""
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

    const int tx = threadIdx.x %% 16;
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
            int r = idx / BK, c = idx %% BK;
            int gr = row0 + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0;
        }
        #pragma unroll
        for (int t = 0; t < (BK * BN) / NTHREADS; t++) {
            int idx = threadIdx.x + t * NTHREADS;
            int r = idx / BN, c = idx %% BN;
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
    print("D1: GEMM 优化扫描（bank conflict + BK）")
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

    a_p = a.data_ptr(); b_p = b.data_ptr()
    c_dev = cu.malloc(N * N * 2)

    print()
    print("  测试 %d³ fp16，配置扫描：" % N)
    print("  %-22s %10s %10s %8s %8s" % ("配置", "耗时(ms)", "TFLOPS", "效率", "max_err"))
    print("  " + "-" * 74)

    results = []
    for BK, PAD in ((16, 0), (32, 0), (32, 4), (64, 4)):
        src = KERNEL % {"BK": BK, "PAD": PAD}
        try:
            ptx = nv.compile(src)
        except Exception as e:
            print("  BK=%-3d PAD=%-2d  COMPILE FAIL: %s" % (BK, PAD, str(e)[:80]))
            continue
        mod, fn = cu.load(ptx)

        # 启动参数
        grid = ((N + 127) // 128, (N + 127) // 128, 1)
        block = (256, 1, 1)
        kernel_params = [ctypes.c_uint64(a_p), ctypes.c_uint64(b_p), ctypes.c_uint64(c_dev.value),
                         ctypes.c_int(N), ctypes.c_int(N), ctypes.c_int(N)]
        param_ptrs = (ctypes.c_void_p * len(kernel_params))(
            *[ctypes.cast(ctypes.byref(pp), ctypes.c_void_p) for pp in kernel_params])

        def launch():
            cu.cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                              0, None, param_ptrs, None)

        # 正确性
        try:
            launch()
            cu.cuCtxSynchronize()
            out = torch.empty(N, N, device="cuda", dtype=torch.float16)
            cu.cuMemcpyDtoH_v2(ctypes.c_void_p(out.data_ptr()), c_dev.value, N * N * 2)
            err = (out.float() - ref).abs().max().item()
        except Exception as e:
            print("  BK=%-3d PAD=%-2d  RUN FAIL: %s" % (BK, PAD, str(e)[:80]))
            cu.cuModuleUnload(mod)
            continue

        # 计时（强制同步，避免 tinygrad 那种异步假象）
        for _ in range(3):
            launch()
        cu.cuCtxSynchronize()
        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            launch()
        cu.cuCtxSynchronize()
        dt = (time.perf_counter() - t0) / iters
        tflops = 2.0 * N**3 / dt / 1e12
        eff = tflops / 3.06 * 100
        label = "BK=%d PAD=%d" % (BK, PAD)
        print("  %-22s %10.3f %10.4f %7.1f%% %8.5f"
              % (label, dt * 1e3, tflops, eff, err))
        results.append((label, tflops, eff, err, BK, PAD))
        cu.cuModuleUnload(mod)

    cu.cuMemFree_v2(c_dev)

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    if results:
        best = max(results, key=lambda x: x[1])
        base = results[0]
        print("  基线 %s : %.4f TFLOPS (%.1f%%)" % (base[0], base[1], base[2]))
        print("  最优 %s : %.4f TFLOPS (%.1f%%)  -> %.2fx"
              % (best[0], best[1], best[2], best[1] / base[1]))
        print("  目标 2.5 TFLOPS (85%%)  达成度 %.0f%%" % (best[1] / 2.5 * 100))
        print()
        print("  参考: cuBLAS fp32 甜区 2.37 TFLOPS / 物理峰值 3.06")
        if best[1] > 2.37:
            print("  >>> 已超过 cuBLAS fp32，自研 kernel 有存在价值")
        else:
            print("  >>> 仍低于 cuBLAS fp32 的 2.37，需继续优化或改用 cuBLAS")

if __name__ == "__main__":
    main()
