#!/usr/bin/env python3
"""
sw_tensor_core_v2.py — 软件 Tensor Core v2：half 输入 + float 累加（无头文件依赖）

关键认识（v1 的教训）：
  `fma.rn.f16x2` 是 FP16 累加。K=2048 时累加器会溢出/精度崩溃，
  所以它只能用于"纯计算峰值"微基准（5.77 TFLOPS），不能做真实 GEMM。

  真实 GEMM 必须用 FP32 累加 -> 那就是 FFMA -> 上限 2.12 TFLOPS（= cuBLAS FP32）。

  但 cuBLAS 的 FP16 路径强制走 HMMA（0.32，因为 TU117 没有真 TC）。
  所以：自定义 kernel（half 输入 + float 累加）能拿到 2.1，
  是 cuBLAS FP16 TENSOR_OP 的 6.6 倍。这就是"软件 TC"的实际价值。

技术路线：
  不用 cuda_fp16.h（NVRTC 找不到），全部用 unsigned short + inline PTX:
    cvt.f32.f16   : half -> float
    cvt.rn.f16.f32: float -> half
  累加用普通 C 的 float FMA（编译器生成 fma.rn.f32）。
"""
import argparse
import ctypes
import glob
import time

import torch


CUDA_SRC = r"""
// 无头文件版本：half 用 unsigned short 表示

__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

#define BM 128
#define BN 128
#define BK 16
#define TM 8     // 每线程 8 行
#define TN 8     // 每线程 8 列
#define NTHREADS 256

extern "C" __global__ void gemm_ffma(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ B,
    unsigned short* __restrict__ C,
    int M, int N, int K)
{
    __shared__ unsigned short As[BM][BK];
    __shared__ unsigned short Bs[BK][BN];

    // 16x16 线程网格，每线程 4x4
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
        // 加载 A tile
        #pragma unroll
        for (int t = 0; t < (BM * BK) / NTHREADS; t++) {
            int idx = threadIdx.x + t * NTHREADS;
            int r = idx / BK, c = idx % BK;
            int gr = row0 + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0;
        }
        // 加载 B tile
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
        # 复用 PyTorch 已创建的 CUDA context，不要新建
        torch.cuda.init()
        torch.zeros(1, device="cuda")   # 确保 context 已建立
        self.cu = ctypes.CDLL("libcuda.so.1")
        rc = self.cu.cuInit(0)
        print("    cuInit rc=%d" % rc)
        self.ctx = ctypes.c_void_p()
        rc = self.cu.cuCtxGetCurrent(ctypes.byref(self.ctx))
        print("    cuCtxGetCurrent rc=%d ctx=%s" % (rc, self.ctx.value))
        if not self.ctx.value:
            dev = ctypes.c_int()
            self.cu.cuDeviceGet(ctypes.byref(dev), 0)
            rc = self.cu.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev)
            print("    cuCtxCreate rc=%d ctx=%s" % (rc, self.ctx.value))
        # 关键：必须显式声明 argtypes，否则 64 位指针会被截断成 int 导致 SIGSEGV
        cu = self.cu
        cu.cuInit.argtypes = [ctypes.c_uint]
        cu.cuInit.restype = ctypes.c_int
        cu.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cu.cuDeviceGet.restype = ctypes.c_int
        cu.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cu.cuCtxGetCurrent.restype = ctypes.c_int
        cu.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        cu.cuCtxCreate_v2.restype = ctypes.c_int
        cu.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        cu.cuModuleLoadData.restype = ctypes.c_int
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        cu.cuModuleGetFunction.restype = ctypes.c_int
        cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        cu.cuMemAlloc_v2.restype = ctypes.c_int
        cu.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        cu.cuMemcpyHtoD_v2.restype = ctypes.c_int
        cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        cu.cuMemcpyDtoH_v2.restype = ctypes.c_int
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                      ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                      ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        cu.cuLaunchKernel.restype = ctypes.c_int
        cu.cuCtxSynchronize.argtypes = []
        cu.cuCtxSynchronize.restype = ctypes.c_int

    def load(self, ptx):
        m = ctypes.c_void_p()
        rc = self.cu.cuModuleLoadData(ctypes.byref(m), ptx.encode() + b"\x00")
        if rc != 0:
            raise RuntimeError("cuModuleLoadData rc=%d" % rc)
        return m

    def func(self, mod, name):
        f = ctypes.c_void_p()
        rc = self.cu.cuModuleGetFunction(ctypes.byref(f), mod, name.encode())
        if rc != 0:
            raise RuntimeError("cuModuleGetFunction rc=%d" % rc)
        return f

    def alloc(self, n):
        p = ctypes.c_uint64()
        self.cu.cuMemAlloc_v2(ctypes.byref(p), n)
        return p

    def h2d(self, dst, src, n):
        self.cu.cuMemcpyHtoD_v2(dst, src, n)

    def d2h(self, src, dst, n):
        self.cu.cuMemcpyDtoH_v2(dst, src, n)

    def launch(self, f, gx, gy, bx, args):
        arr = (ctypes.c_void_p * len(args))()
        keep = list(args)
        for i, a in enumerate(args):
            arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
        rc = self.cu.cuLaunchKernel(f, gx, gy, 1, bx, 1, 1, 0, None,
                                    ctypes.cast(arr, ctypes.c_void_p), None)
        if rc != 0:
            raise RuntimeError("cuLaunchKernel rc=%d" % rc)

    def sync(self):
        self.cu.cuCtxSynchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048)
    args = ap.parse_args()

    print("=" * 74)
    print("软件 Tensor Core v2 — half 输入 + float 累加（无头文件）")
    print("=" * 74)
    p = torch.cuda.get_device_properties(0)
    print("GPU: %s  SM=%d  CC=%d.%d" % (p.name, p.multi_processor_count, p.major, p.minor))
    print()

    print("[1] NVRTC 编译 ...")
    nv = Nvrtc()
    ptx = nv.compile(CUDA_SRC, "sm_75")
    print("    PTX %d 字符" % len(ptx))
    print("    含 cvt.f32.f16: %s | 含 fma.rn.f32: %s | 含 mma.sync: %s"
          % ("cvt.f32.f16" in ptx, "fma.rn.f32" in ptx, "mma.sync" in ptx))
    with open("/tmp/sw_tc2.ptx", "w") as f:
        f.write(ptx)
    print()

    print("[2] 加载 ...")
    cu = Cuda()
    mod = cu.load(ptx)
    fn = cu.func(mod, "gemm_ffma")
    print()

    M = N = K = args.size
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    nA, nB, nC = a.numel() * 2, b.numel() * 2, c.numel() * 2
    dA, dB, dC = cu.alloc(nA), cu.alloc(nB), cu.alloc(nC)
    cu.h2d(dA, a.data_ptr(), nA)
    cu.h2d(dB, b.data_ptr(), nB)
    cu.h2d(dC, c.data_ptr(), nC)

    BM = BN = 128
    gx, gy = (N + BN - 1) // BN, (M + BM - 1) // BM
    argv = [ctypes.c_uint64(dA.value), ctypes.c_uint64(dB.value),
            ctypes.c_uint64(dC.value),
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]

    print("[3] 正确性 ...")
    cu.launch(fn, gx, gy, 256, argv)
    cu.sync()
    cu.d2h(dC, c.data_ptr(), nC)
    ref = (a.float() @ b.float()).half()
    err = (c.float() - ref.float()).abs().max().item()
    rel = err / ref.float().abs().max().item()
    print("    max abs err = %.5f  (相对 %.2e)" % (err, rel))
    print()

    print("[4] 测速 ...")
    iters = 10
    cu.launch(fn, gx, gy, 256, argv)
    cu.sync()
    t0 = time.time()
    for _ in range(iters):
        cu.launch(fn, gx, gy, 256, argv)
    cu.sync()
    dt = (time.time() - t0) / iters
    tf = 2.0 * M * N * K / dt / 1e12
    print("    %.3f TFLOPS   (%.1f ms/次)" % (tf, dt * 1000))
    print()
    print("=" * 74)
    print("对比（本机实测）:")
    print("  cuBLAS TENSOR_OP (HMMA 模拟) : 0.320 TFLOPS  ->  %.1fx" % (tf / 0.32))
    print("  cuBLAS FP32                  : 2.122 TFLOPS  ->  %.2fx" % (tf / 2.122))
    print("  FP16x2 纯计算峰值(FP16累加)   : 5.770 TFLOPS  (不可用于真实 GEMM)")
    print("=" * 74)


if __name__ == "__main__":
    main()
