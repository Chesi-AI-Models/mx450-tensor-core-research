# -*- coding: utf-8 -*-
"""D1 天花板真相核查 —— 「3.06 TFLOPS」这个基准到底对不对？

疑点: 全程 nvidia-smi 报 clocks.sm = 1395 MHz，而不是记忆文件里用的 1665 MHz。
      若真实时钟是 1395，则物理峰值 = 2*64*14*1.395 = 2.498 TFLOPS，
      我们已测到 2.4872 (C7) = 99.6%，「85% 目标」这个基准本身就是错的。

本脚本三件事（全部实测，不查参数表）:
 1. 打印 nvidia-smi 的图形/SM 时钟范围（Min/Max/Current）
 2. 纯 FP32 FFMA 微基准 —— 用运行时才知的系数防止编译器折叠，测真实峰值
 3. 用实测峰值重算所有历史效率
"""
import ctypes, glob, re, subprocess, time
import torch


def _find(patterns):
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


def clocks():
    out = {}
    try:
        txt = subprocess.run(["nvidia-smi", "-q", "-d", "CLOCK"],
                             capture_output=True, text=True, timeout=15).stdout
        for key in ("Graphics", "SM"):
            m = re.search(r"%s\s*:\s*(\d+)\s*MHz" % key, txt)
            if m:
                out["current_" + key] = int(m.group(1))
        # Max Clocks 段
        seg = txt.split("Max Clocks")[-1] if "Max Clocks" in txt else ""
        for key in ("Graphics", "SM"):
            m = re.search(r"%s\s*:\s*(\d+)\s*MHz" % key, seg)
            if m:
                out["max_" + key] = int(m.group(1))
        seg2 = txt.split("Clocks Event Reasons")[0]
        m = re.findall(r"(\d+)\s*MHz", seg2)
        out["sample"] = ",".join(m[:8])
    except Exception as e:
        out["err"] = str(e)
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,"
                            "clocks.gr,clocks.max.gr,temperature.gpu,power.draw,"
                            "power.limit,enforced.power.limit",
                            "--format=csv,noheader"], capture_output=True, text=True,
                           timeout=10).stdout.strip()
        out["query"] = q
    except Exception as e:
        out["query"] = "err %s" % e
    return out


class Nvrtc:
    def __init__(self):
        self.lib = ctypes.CDLL(_find([
            "/vol1/ganlin/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
            "/usr/lib/x86_64-linux-gnu/libnvrtc.so*"]))

    def compile(self, src, arch="sm_75"):
        prog = ctypes.c_void_p()
        if self.lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"f.cu", 0, None, None):
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
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        cu.cuCtxSynchronize.argtypes = []
        cu.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                          ctypes.c_void_p]

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

    def regs(self, fn):
        r = ctypes.c_int()
        self.cu.cuFuncGetAttribute(ctypes.byref(r), 4, fn)
        return r.value


# 32 条独立累加链，系数在运行时由 threadIdx 算出 -> 编译器无法折叠
FMA = r"""
#define NACC 32
#define ITER __ITER__
extern "C" __global__ void fma_peak(const float* __restrict__ in, float* __restrict__ out,
                                    int n, int dummy)
{
    float x = in[threadIdx.x & 31] + 1.0000001f;   // 运行时值
    float y = in[(threadIdx.x + 7) & 31] * 1e-7f;
    float a[NACC];
    #pragma unroll
    for (int i = 0; i < NACC; i++) a[i] = in[(threadIdx.x + i) & 31];
    #pragma unroll 1
    for (int t = 0; t < ITER; t++) {
        #pragma unroll
        for (int i = 0; i < NACC; i++) a[i] = fmaf(a[i], x, y);
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < NACC; i++) s += a[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
"""

# 双精度版本（FP64 单元只有 1/32，用来交叉验证时钟）
FMA64 = r"""
#define NACC 16
#define ITER __ITER__
extern "C" __global__ void fma_peak(const float* __restrict__ in, float* __restrict__ out,
                                    int n, int dummy)
{
    double x = (double)in[threadIdx.x & 31] + 1.0000001;
    double y = (double)in[(threadIdx.x + 7) & 31] * 1e-7;
    double a[NACC];
    #pragma unroll
    for (int i = 0; i < NACC; i++) a[i] = (double)in[(threadIdx.x + i) & 31];
    #pragma unroll 1
    for (int t = 0; t < ITER; t++) {
        #pragma unroll
        for (int i = 0; i < NACC; i++) a[i] = fma(a[i], x, y);
    }
    double s = 0.;
    #pragma unroll
    for (int i = 0; i < NACC; i++) s += a[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = (float)s;
}
"""


def bench_fma(cu, nv, src_tpl, iters_loop, nblocks=14 * 4, nthreads=256, nacc=32):
    src = src_tpl.replace("__ITER__", str(iters_loop))
    ptx = nv.compile(src)
    mod, fn = cu.load(ptx, b"fma_peak")
    inbuf = torch.randn(64, device="cuda", dtype=torch.float32).abs() + 1.0
    outbuf = torch.zeros(nblocks * nthreads, device="cuda", dtype=torch.float32)
    kp = [ctypes.c_uint64(inbuf.data_ptr()), ctypes.c_uint64(outbuf.data_ptr()),
          ctypes.c_int(0), ctypes.c_int(0)]
    pp = (ctypes.c_void_p * 4)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in kp])

    def launch():
        cu.cuLaunchKernel(fn, nblocks, 1, 1, nthreads, 1, 1, 0, None, pp, None)

    launch()
    cu.cuCtxSynchronize()
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        launch()
    cu.cuCtxSynchronize()
    dt = (time.perf_counter() - t0) / reps
    flop = 2.0 * nacc * iters_loop * nthreads * nblocks
    return flop / dt / 1e12, cu.regs(fn)


def main():
    print("=" * 92)
    print("D1 天花板真相核查")
    print("=" * 92)
    c0 = clocks()
    print("  nvidia-smi 时钟原始输出:")
    for k, v in c0.items():
        print("    %-14s %s" % (k, v))

    cu, nv = Cuda(), Nvrtc()
    print()
    print("  测量中…")
    tf32, regs32 = bench_fma(cu, nv, FMA, 20000, nacc=32)
    tf64, regs64 = bench_fma(cu, nv, FMA64, 20000, nacc=16)
    c1 = clocks()
    print("  测后时钟:", c1.get("query"))

    print()
    print("=" * 92)
    print("  实测结果")
    print("=" * 92)
    print("  纯 FP32 FFMA 峰值  : %.4f TFLOPS  (regs=%d)" % (tf32, regs32))
    print("  纯 FP64 FMA  峰值  : %.4f TFLOPS  (regs=%d)  注: TU117 FP64 = FP32 的 1/32" % (tf64, regs64))
    print()
    print("  反解时钟:")
    print("    由 FP32 反解: %.1f MHz  (2*64*14*clk)" % (tf32 / (2 * 64 * 14) * 1e6))
    print("    由 FP64 反解: %.1f MHz  (2*2*14*clk，TU117 每 SM 2 个 FP64 单元)"
          % (tf64 / (2 * 2 * 14) * 1e6))
    print()
    print("  历史数据重算（用实测峰值 %.4f 作分母）:" % tf32)
    hist = [("v2 初版", 1.603), ("v2+v4 转置", 1.876), ("v5 float4 全局", 1.9714),
            ("v7 S5", 2.3075), ("C7 64x128x8 P0 单", 2.4872), ("C8 128x128x8 P16 双", 2.4851),
            ("cuBLAS fp32 mm", 2.1333)]
    for name, v in hist:
        print("    %-22s %.4f  = 实测峰值的 %.1f%%" % (name, v, v / tf32 * 100))
    print()
    print("  -> 若以上百分比接近 100%，说明「3.06 TFLOPS」这个 1665MHz 假设是错的，")
    print("     D1「52%%→85%%」的目标应改为「相对真实峰值的百分比」。")


if __name__ == "__main__":
    main()
