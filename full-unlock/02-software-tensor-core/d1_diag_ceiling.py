# -*- coding: utf-8 -*-
"""D1 诊断: 为什么停在 64%？—— 寄存器 / 占用率 / 纯 FFMA 天花板

实测事实（不许猜）:
  v5 V6  半精度smem + float4 全局加载  BK=16 PAD=8  -> 1.9714 TFLOPS (64.4%)
  v6 W4  float smem + A 转置 + LDS.128  BK=16 PADT=12 -> 1.8113 TFLOPS (59.2%)  <<< 更慢！

  说明「减少内层指令数」不是当前瓶颈。本脚本测三件事:
   1. 每个 kernel 的实际寄存器数 / 每 SM 活跃 block 数（占用率）
   2. 纯 FFMA 天花板: 同样 8x8 acc、256 线程，但内层只做 FFMA 不读 smem
      -> 这个数字才是「本 launch 配置能达到的上限」
   3. acc 尺寸扫描(4x4 / 8x8 / 8x4): 寄存器压力 vs 数据复用
"""
import ctypes, glob, time
import torch


def _find(patterns):
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


NVCC_ATTR = {"MAX_THREADS_PER_BLOCK": 0, "NUM_REGS": 4, "SHARED_SIZE_BYTES": 8,
             "MAX_THREADS_PER_BLOCK_": 1}


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

    def load(self, ptx, name=b"gemm_ffma"):
        mod = ctypes.c_void_p()
        rc = self.cu.cuModuleLoadData(ctypes.byref(mod), ptx.encode())
        if rc:
            raise RuntimeError("load rc=%d" % rc)
        fn = ctypes.c_void_p()
        rc = self.cu.cuModuleGetFunction(ctypes.byref(fn), mod, name)
        if rc:
            raise RuntimeError("getfn rc=%d" % rc)
        return mod, fn

    def attrs(self, fn, nthreads):
        regs = ctypes.c_int()
        self.cu.cuFuncGetAttribute(ctypes.byref(regs), 4, fn)   # NUM_REGS
        smem = ctypes.c_int()
        self.cu.cuFuncGetAttribute(ctypes.byref(smem), 8, fn)   # SHARED_SIZE_BYTES
        blocks = ctypes.c_int()
        self.cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(blocks), fn, nthreads, 0)
        return regs.value, smem.value, blocks.value

    def malloc(self, nbytes):
        p = ctypes.c_uint64()
        rc = self.cu.cuMemAlloc_v2(ctypes.byref(p), nbytes)
        if rc:
            raise RuntimeError("malloc rc=%d" % rc)
        return p


SM = 14
CLK = 1.665e9

# ---------------------------------------------------------------- 纯 FFMA 天花板
FFMA_SRC = r"""
#define TM __TM__
#define TN __TN__
extern "C" __global__ void gemm_ffma(const unsigned short* A, const unsigned short* B,
                                     unsigned short* C, int M, int N, int K)
{
    float a[TM], b[TN];
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) { a[i] = A[threadIdx.x + i] * 1e-5f; }
    #pragma unroll
    for (int j = 0; j < TN; j++) { b[j] = B[threadIdx.x + j] * 1e-5f; }
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    int reps = K / (TM * TN);          // 保证总 FFMA 次数与 N,K 对应
    for (int r = 0; r < reps; r++) {
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j++)
                acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        #pragma unroll
        for (int i = 0; i < TM; i++) a[i] = a[i] + 1e-30f;
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) s += acc[i][j];
    if (s == 1234.5f) C[threadIdx.x] = (unsigned short)s;
}
"""


def run_ffma(cu, nv, TM, TN, nblocks, nthreads, K):
    src = FFMA_SRC.replace("__TM__", str(TM)).replace("__TN__", str(TN))
    ptx = nv.compile(src)
    mod, fn = cu.load(ptx)
    regs, smem, blocks = cu.attrs(fn, nthreads)
    a = torch.zeros(1024, device="cuda", dtype=torch.float16)
    b = torch.zeros(1024, device="cuda", dtype=torch.float16)
    c = torch.zeros(1024, device="cuda", dtype=torch.float16)
    kp = [ctypes.c_uint64(a.data_ptr()), ctypes.c_uint64(b.data_ptr()),
          ctypes.c_uint64(c.data_ptr()), ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(K)]
    pp = (ctypes.c_void_p * 6)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in kp])

    def launch():
        cu.cuLaunchKernel(fn, nblocks, 1, 1, nthreads, 1, 1, 0, None, pp, None)

    launch(); cu.cuCtxSynchronize()
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        launch()
    cu.cuCtxSynchronize()
    dt = (time.perf_counter() - t0) / iters
    flop = 2.0 * TM * TN * K * nthreads * nblocks
    tf = flop / dt / 1e12
    cu.cuModuleUnload(mod)
    return tf, regs, blocks


def main():
    print("=" * 88)
    print("D1 诊断: 寄存器 / 占用率 / 纯 FFMA 天花板")
    print("=" * 88)
    torch.cuda.init()
    p = torch.cuda.get_device_properties(0)
    print("  GPU: %s  SM=%d  clock=%.3f GHz  regs/SM=65536  smem/block(max)=48KB+"
          % (p.name, p.multi_processor_count, CLK / 1e9))
    print("  理论 FFMA 峰值: 2*64*%d*%.3f GHz = %.3f TFLOPS"
          % (p.multi_processor_count, CLK / 1e9,
             2 * 64 * p.multi_processor_count * CLK / 1e12))

    cu = Cuda()
    nv = Nvrtc()

    print()
    print("【1】纯 FFMA 天花板（无 smem 读取，只有 FMA 依赖链）")
    print("  %-26s %9s %6s %8s %8s" % ("配置", "TFLOPS", "效率", "regs", "blocks/SM"))
    print("  " + "-" * 70)
    # 与真实 GEMM 同规模: N=2048 -> 总 FLOP 1.72e10
    target_blocks = ((2048 + 127) // 128) ** 2
    for TM, TN, nb, nt in [(4, 4, target_blocks, 256),
                           (8, 8, target_blocks, 256),
                           (8, 4, target_blocks, 256),
                           (8, 8, target_blocks, 128),
                           (16, 8, target_blocks, 128)]:
        K = int(1.72e10 / (2.0 * TM * TN * nt * nb))
        K = max(64, (K // (TM * TN)) * (TM * TN))
        tf, regs, blocks = run_ffma(cu, nv, TM, TN, nb, nt, K)
        print("  TM%-2d TN%-2d nt=%-3d K=%-6d %9.4f %5.1f%% %8d %8d"
              % (TM, TN, nt, K, tf, tf / 3.06 * 100, regs, blocks))

    print()
    print("【2】结论对照")
    print("  实测最优 GEMM : 1.9714 TFLOPS (64.4%)")
    print("  -> 若纯 FFMA 天花板 ≈ 3.0，则差距来自 smem 访存/依赖，不是寄存器")
    print("  -> 若纯 FFMA 天花板 << 3.0，则本 launch 配置根本到不了 85%，必须换 tile/线程")


if __name__ == "__main__":
    main()
