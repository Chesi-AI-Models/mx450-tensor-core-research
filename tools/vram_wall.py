#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MX450 显存边界实战测试
实验 A：逐级分配，精确测出显存天花板（能否突破 2GB）
实验 B：Unified Memory 伪装扩容 —— 让它"看到" 4GB，测真实性能代价
"""
import ctypes, time

cu = ctypes.CDLL("libcuda.so.1")
def ck(r, n=""):
    return r
def chk(r, n=""):
    if r != 0: raise RuntimeError("CUDA err %d @ %s" % (r, n))

cu.cuInit.argtypes=[ctypes.c_uint]; chk(cu.cuInit(0),"init")
d=ctypes.c_int(); chk(cu.cuDeviceGet(ctypes.byref(d),0),"dev")
c=ctypes.c_void_p(); chk(cu.cuCtxCreate_v2(ctypes.byref(c),0,d),"ctx")

def attr(a):
    v=ctypes.c_int(); cu.cuDeviceGetAttribute(ctypes.byref(v),a,d); return v.value

TOTAL   = attr(65)   # TOTAL_MEMORY
FREE    = attr(66)   # FREE_MEMORY (context 内可用)
MANAGED = attr(83)   # MANAGED_MEMORY 支持
CONCUR  = attr(89)   # CONCURRENT_MANAGED_ACCESS
UNIFIED = attr(18)   # UNIFIED_ADDRESSING

print("=" * 64)
print("设备属性")
print("  TOTAL_MEMORY           : %.0f MiB" % (TOTAL/1048576))
print("  FREE_MEMORY            : %.0f MiB" % (FREE/1048576))
print("  MANAGED_MEMORY 支持     : %s" % ("是" if MANAGED else "否"))
print("  CONCURRENT_MANAGED     : %s" % ("是" if CONCUR else "否"))
print("  UNIFIED_ADDRESSING     : %s" % ("是" if UNIFIED else "否"))
print("=" * 64)

cu.cuMemAlloc_v2.argtypes=[ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
cu.cuMemFree_v2.argtypes=[ctypes.c_uint64]
cu.cuMemsetD8_v2.argtypes=[ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t]

print("\n【实验 A】逐级分配普通显存（每档都 memset 压实）")
print("-" * 64)
ok_list, fail_list = [], []
for mb in (512, 1024, 1400, 1500, 1600, 1620, 1640, 1660, 1680, 1700, 1750, 1800, 1900, 2048, 2500, 3072, 4096):
    p = ctypes.c_uint64()
    r = cu.cuMemAlloc_v2(ctypes.byref(p), mb*1024*1024)
    if r == 0:
        rc = cu.cuMemsetD8_v2(p, 0xA5, mb*1024*1024)
        cu.cuMemFree_v2(p)
        if rc == 0:
            ok_list.append(mb); print("  %5d MiB  ✅ 成功（含写入）" % mb)
        else:
            fail_list.append(mb); print("  %5d MiB  ❌ 分配成功但写入失败 (err %d)" % (mb, rc))
    else:
        fail_list.append(mb); print("  %5d MiB  ❌ 失败 (err %d)" % (mb, r))

print("-" * 64)
print("  最大可分配: %d MiB" % (max(ok_list) if ok_list else 0))
print("  首次失败  : %d MiB" % (min(fail_list) if fail_list else -1))

# ---- 实验 B: Unified Memory 伪装扩容 ----
print("\n【实验 B】Unified Memory 超配（伪装扩容）")
print("-" * 64)
cu.cuMemAllocManaged.argtypes=[ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_uint]
TARGET_GB = 4
bytes_needed = TARGET_GB * 1024**3
p = ctypes.c_uint64()
r = cu.cuMemAllocManaged(ctypes.byref(p), bytes_needed, 1)  # 1 = CU_MEM_ATTACH_GLOBAL
if r != 0:
    print("  cuMemAllocManaged(%dGB) 失败 err=%d" % (TARGET_GB, r))
else:
    print("  cuMemAllocManaged(%dGB) 分配成功 —— 显存只有 2GB，它却给了 %dGB" % (TARGET_GB, TARGET_GB))
    print("  现在写入并测速（数据会被迫换页到主机内存）...")
    CHUNK = 256 * 1024 * 1024
    t0 = time.time()
    written = 0
    for off in range(0, bytes_needed, CHUNK):
        cu.cuMemsetD8_v2(p.value + off, 0x5A, min(CHUNK, bytes_needed - off))
        written += min(CHUNK, bytes_needed - off)
    chk(cu.cuCtxSynchronize(), "sync")
    dt = time.time() - t0
    print("  写入 %.1f GB 耗时 %.2f s  → 有效带宽 %.2f GB/s" % (written/1e9, dt, written/dt/1e9))
    print("  对比：显存内带宽实测 69.3 GB/s")
    print("  性能损失倍数：%.1f×" % (69.3 / (written/dt/1e9)))
    cu.cuMemFree_v2(p)
print("=" * 64)
print("DONE")
