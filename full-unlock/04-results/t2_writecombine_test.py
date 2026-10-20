# -*- coding: utf-8 -*-
"""T2-3 续: READ_ONLY 被驱动拒绝后，改用两条正路实现「避免回写」

上一轮实测:
  CU_MEMHOSTREGISTER_DEVICEMAP | READ_ONLY  -> ret=1 (CUDA_ERROR_INVALID_VALUE)
  普通 DEVICEMAP      H2D 3.15 GB/s  D2H 3.29 GB/s
  PORTABLE            H2D 3.15 GB/s  D2H 3.29 GB/s

本脚本测两条替代路径:
  路径 1  cuMemHostAlloc + CU_MEMHOSTALLOC_WRITE_COMBINED (0x04)
          —— 写合并内存，只对 host 写、device 读的单向流有利，
             正是「权重常驻 host、GPU 反复读」的场景
  路径 2  cuMemAllocManaged + cuMemAdvise(CU_MEM_ADVISE_SET_READ_MOSTLY=1)
          —— 这才是 MEMORY.md 里写的 cuMemAdviseSetReadMostly 本体
             也是 GreenBoost shim 的 Path C（last resort）

对照: 普通 cuMemHostAlloc(DEVICEMAP)
"""
import ctypes, mmap, time

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
cuMemHostAlloc = bind("cuMemHostAlloc", ctypes.c_int,
                      [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint])
cuMemFreeHost = bind("cuMemFreeHost", ctypes.c_int, [ctypes.c_void_p])
cuMemHostGetDevicePointer_v2 = bind("cuMemHostGetDevicePointer_v2", ctypes.c_int,
                                    [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint])
cuMemAllocManaged = bind("cuMemAllocManaged", ctypes.c_int,
                         [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_uint])
cuMemAdvise = bind("cuMemAdvise", ctypes.c_int,
                   [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int, ctypes.c_int])
cuMemcpyHtoD_v2 = bind("cuMemcpyHtoD_v2", ctypes.c_int,
                       [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t])
cuMemcpyDtoH_v2 = bind("cuMemcpyDtoH_v2", ctypes.c_int,
                       [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t])
cuCtxSynchronize = bind("cuCtxSynchronize", ctypes.c_int, [])

CU_MEMHOSTALLOC_PORTABLE = 0x01
CU_MEMHOSTALLOC_DEVICEMAP = 0x02
CU_MEMHOSTALLOC_WRITE_COMBINED = 0x04
CU_MEM_ATTACH_GLOBAL = 0x01
CU_MEM_ADVISE_SET_READ_MOSTLY = 1

import torch
torch.cuda.init()
_x = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
ctx = ctypes.c_void_p()
cuCtxGetCurrent(ctypes.byref(ctx))

print("=" * 74)
print("T2-3 续: WRITE_COMBINED vs MANAGED+READ_MOSTLY")
print("=" * 74)
print("  ctx =", hex(ctx.value or 0))

SIZE = 256 << 20
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

# ---------- 路径 0: 对照 cuMemHostAlloc(DEVICEMAP)
def path_hostalloc(label, flags):
    pptr = ctypes.c_void_p()
    r = cuMemHostAlloc(ctypes.byref(pptr), SIZE, flags)
    if r != 0:
        print("  %-34s cuMemHostAlloc ret=%d FAIL" % (label, r))
        return None
    dptr = ctypes.c_uint64()
    r2 = cuMemHostGetDevicePointer_v2(ctypes.byref(dptr), pptr, 0)
    if r2 != 0:
        print("  %-34s devptr ret=%d FAIL" % (label, r2))
        cuMemFreeHost(pptr)
        return None
    dbuf = ctypes.c_uint64()
    cuMemAlloc_v2(ctypes.byref(dbuf), SIZE)
    h2d = bench(lambda: cuMemcpyHtoD_v2(dbuf, pptr, SIZE))
    d2h = bench(lambda: cuMemcpyDtoH_v2(pptr, dbuf, SIZE))
    cuMemFree_v2(dbuf)
    cuMemFreeHost(pptr)
    print("  %-34s flags=0x%02X H2D %6.2f GB/s  D2H %6.2f GB/s"
          % (label, flags, SIZE / h2d / 1e9, SIZE / d2h / 1e9))
    return (SIZE / h2d / 1e9, SIZE / d2h / 1e9)

# ---------- 路径 2: managed + READ_MOSTLY
def path_managed(label, readmostly):
    dptr = ctypes.c_uint64()
    r = cuMemAllocManaged(ctypes.byref(dptr), SIZE, CU_MEM_ATTACH_GLOBAL)
    if r != 0:
        print("  %-34s cuMemAllocManaged ret=%d FAIL" % (label, r))
        return None
    if readmostly:
        ra = cuMemAdvise(dptr, SIZE, CU_MEM_ADVISE_SET_READ_MOSTLY, 0)
        if ra != 0:
            print("  %-34s cuMemAdvise ret=%d FAIL" % (label, ra))
    # managed 内存的 host 指针就是 dptr 本身
    hptr = ctypes.c_void_p(dptr.value)
    dbuf = ctypes.c_uint64()
    cuMemAlloc_v2(ctypes.byref(dbuf), SIZE)
    try:
        h2d = bench(lambda: cuMemcpyHtoD_v2(dbuf, hptr, SIZE))
        d2h = bench(lambda: cuMemcpyDtoH_v2(hptr, dbuf, SIZE))
        print("  %-34s %-8s H2D %6.2f GB/s  D2H %6.2f GB/s"
              % (label, "RO" if readmostly else "normal",
                 SIZE / h2d / 1e9, SIZE / d2h / 1e9))
        res = (SIZE / h2d / 1e9, SIZE / d2h / 1e9)
    except Exception as e:
        print("  %-34s bench FAIL %s" % (label, e))
        res = None
    cuMemFree_v2(dbuf)
    cuMemFree_v2(dptr)
    return res

print()
print("-" * 74)
print("路径 1: cuMemHostAlloc")
print("-" * 74)
base = path_hostalloc("对照 DEVICEMAP", CU_MEMHOSTALLOC_DEVICEMAP)
wc = path_hostalloc("WRITE_COMBINED", CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITE_COMBINED)
wc2 = path_hostalloc("PORTABLE|WRITE_COMBINED", CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_WRITE_COMBINED)

print()
print("-" * 74)
print("路径 2: cuMemAllocManaged + cuMemAdvise")
print("-" * 74)
mg = path_managed("MANAGED", False)
mr = path_managed("MANAGED + SET_READ_MOSTLY", True)

print()
print("=" * 74)
print("结论")
print("=" * 74)
def cmp(name, a, b):
    if a and b:
        print("  %-32s H2D %5.2fx  D2H %5.2fx" % (name, b[0] / a[0], b[1] / a[1]))

if base and wc:
    cmp("WRITE_COMBINED vs 普通", base, wc)
if base and wc2:
    cmp("PORTABLE|WC vs 普通", base, wc2)
if mg and mr:
    cmp("READ_MOSTLY vs 普通managed", mg, mr)
print()
print("  参照: MEMORY.md 记录的 T2 带宽 5.65 GB/s / VRAM 67.52 GB/s")
