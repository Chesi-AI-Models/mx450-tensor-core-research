# -*- coding: utf-8 -*-
"""T2-3 实测: CU_MEMHOSTREGISTER_READ_ONLY 能否减少回写、提升 H2D 带宽

背景（MEMORY.md L82-85 的待验证方向之一）:
  "cuMemAdviseSetReadMostly（避免回写）"

但 GreenBoost 主路径不是 managed memory，而是 pinned host memory:
    cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_DEVICEMAP)   # 0x02
所以真正等价的手段是 **CU_MEMHOSTREGISTER_READ_ONLY (0x04)** ——
CUDA 11.0+ 提供，告诉驱动这块 host 内存不会被 GPU 写，省掉 cache 一致性维护。

本脚本用 ctypes 直接调 CUDA Driver API，对比三种配置的实测带宽:
  A. 普通 pinned（DEVICEMAP=0x02）        <- GreenBoost 现状
  B. 只读 pinned（DEVICEMAP|READ_ONLY=0x06）
  C. portable（PORTABLE=0x01）作为对照
"""
import ctypes, time, sys

# ---- CUDA Driver API 绑定（必须显式声明 argtypes，否则 64 位指针截断）
cuda = ctypes.CDLL("libcuda.so.1")

def bind(name, restype, argtypes):
    fn = getattr(cuda, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn

cuInit = bind("cuInit", ctypes.c_int, [ctypes.c_uint])
cuCtxGetCurrent = bind("cuCtxGetCurrent", ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)])
cuMemAlloc_v2 = bind("cuMemAlloc_v2", ctypes.c_int,
                     [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t])
cuMemFree_v2 = bind("cuMemFree_v2", ctypes.c_int, [ctypes.c_uint64])
cuMemHostRegister_v2 = bind("cuMemHostRegister_v2", ctypes.c_int,
                            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint])
cuMemHostUnregister = bind("cuMemHostUnregister", ctypes.c_int, [ctypes.c_void_p])
cuMemHostGetDevicePointer_v2 = bind("cuMemHostGetDevicePointer_v2", ctypes.c_int,
                                    [ctypes.POINTER(ctypes.c_uint64),
                                     ctypes.c_void_p, ctypes.c_uint])
cuMemcpyHtoD_v2 = bind("cuMemcpyHtoD_v2", ctypes.c_int,
                       [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t])
cuMemcpyDtoH_v2 = bind("cuMemcpyDtoH_v2", ctypes.c_int,
                       [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t])
cuCtxSynchronize = bind("cuCtxSynchronize", ctypes.c_int, [])

CU_MEMHOSTREGISTER_PORTABLE = 0x01
CU_MEMHOSTREGISTER_DEVICEMAP = 0x02
CU_MEMHOSTREGISTER_READ_ONLY = 0x04     # CUDA 11.0+

# ---- 初始化 context（必须复用 torch 的，且要真正变 current）
import torch
torch.cuda.init()
# 关键：必须跑一次真实的 CUDA 操作，context 才会变成当前线程的 current context
_x = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
ctx = ctypes.c_void_p()
r = cuCtxGetCurrent(ctypes.byref(ctx))
if not ctx.value:
    # 兜底：用 primary context
    cuDevicePrimaryCtxRetain = bind("cuDevicePrimaryCtxRetain", ctypes.c_int,
                                    [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int])
    cuCtxSetCurrent = bind("cuCtxSetCurrent", ctypes.c_int, [ctypes.c_void_p])
    pc = ctypes.c_void_p()
    rr = cuDevicePrimaryCtxRetain(ctypes.byref(pc), 0)
    print("  primary ctx retain ret=%d ctx=%s" % (rr, hex(pc.value or 0)))
    if pc.value:
        cuCtxSetCurrent(pc)
        r = cuCtxGetCurrent(ctypes.byref(ctx))
print("=" * 72)
print("T2-3: CU_MEMHOSTREGISTER_READ_ONLY 实测")
print("=" * 72)
print("  torch", torch.__version__, "| cuda ctx =", hex(ctx.value or 0), "| ret", r)
p = torch.cuda.get_device_properties(0)
print("  GPU:", p.name, "CC=%d.%d" % (p.major, p.minor))

SIZE = 256 << 20        # 256 MB
ITERS = 12

def bench(fn, iters=ITERS, warmup=2):
    for _ in range(warmup):
        fn()
    cuCtxSynchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    cuCtxSynchronize()
    return (time.perf_counter() - t0) / iters

def test(label, flags):
    import mmap
    # 用 mmap 拿页对齐的 host 内存（与 GreenBoost 同款做法）
    buf = mmap.mmap(-1, SIZE)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    r = cuMemHostRegister_v2(ctypes.c_void_p(addr), SIZE, flags)
    if r != 0:
        print("  %-28s cuMemHostRegister ret=%d FAIL" % (label, r))
        buf.close()
        return None
    dptr = ctypes.c_uint64()
    r2 = cuMemHostGetDevicePointer_v2(ctypes.byref(dptr), ctypes.c_void_p(addr), 0)
    if r2 != 0:
        print("  %-28s cuMemHostGetDevicePointer ret=%d FAIL" % (label, r2))
        cuMemHostUnregister(ctypes.c_void_p(addr))
        buf.close()
        return None

    # device 端缓冲
    dbuf = ctypes.c_uint64()
    cuMemAlloc_v2(ctypes.byref(dbuf), SIZE)

    h2d = bench(lambda: cuMemcpyHtoD_v2(dbuf, ctypes.c_void_p(addr), SIZE))
    d2h = bench(lambda: cuMemcpyDtoH_v2(ctypes.c_void_p(addr), dbuf, SIZE))

    cuMemFree_v2(dbuf)
    cuMemHostUnregister(ctypes.c_void_p(addr))
    buf.close()

    hb = SIZE / h2d / 1e9
    db = SIZE / d2h / 1e9
    print("  %-28s flags=0x%02X  H2D %7.3f ms (%6.2f GB/s)  D2H %7.3f ms (%6.2f GB/s)"
          % (label, flags, h2d * 1e3, hb, d2h * 1e3, db))
    return hb, db

print()
print("-" * 72)
print("测试 %d MB, %d 次迭代" % (SIZE >> 20, ITERS))
print("-" * 72)

resA = test("A. 普通 DEVICEMAP", CU_MEMHOSTREGISTER_DEVICEMAP)
resB = test("B. 只读 DEVICEMAP|RO", CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_READ_ONLY)
resC = test("C. PORTABLE (对照)", CU_MEMHOSTREGISTER_PORTABLE)
resD = test("D. PORTABLE|RO (对照)", CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_READ_ONLY)

print()
print("=" * 72)
print("结论")
print("=" * 72)
if resA and resB:
    print("  H2D: 只读比普通 %.2fx  (%6.2f -> %6.2f GB/s)"
          % (resB[0] / resA[0], resA[0], resB[0]))
    print("  D2H: 只读比普通 %.2fx  (%6.2f -> %6.2f GB/s)"
          % (resB[1] / resA[1], resA[1], resB[1]))
    gain = resB[0] / resA[0]
    if gain > 1.05:
        print()
        print("  >>> READ_ONLY 有显著收益 (%.0f%%)，值得改进 GreenBoost shim" % ((gain - 1) * 100))
    elif gain > 1.01:
        print()
        print("  >>> READ_ONLY 有轻微收益 (%.1f%%)" % ((gain - 1) * 100))
    else:
        print()
        print("  >>> READ_ONLY 无明显收益 —— MX450/PCIe3 上该标志不影响传输")
