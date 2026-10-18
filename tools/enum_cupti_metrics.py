#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
枚举 MX450 (TU117) 通过 CUPTI 暴露的全部性能计数器（metrics）。

判定逻辑：
  - 若出现 sm__pipe_tensor* / *inst_executed_pipe_tensor* 类计数器
      → 硬件存在张量管线，且可被采样  → TC 单元真实存在
  - 若完全没有 tensor 相关计数器
      → 硬件层面就没有这条通路        → 更接近「物理不存在」

全程只读：仅查询计数器名称与元数据，不启动任何采集、不修改 GPU。
"""
import ctypes, sys

CUPTI_PATH = "/vol1/ganlin/venv/lib/python3.11/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12"

cupti = ctypes.CDLL(CUPTI_PATH)
cuda = ctypes.CDLL("libcuda.so.1")

# ---- CUDA 初始化 ----
cuda.cuInit.argtypes = [ctypes.c_uint]
rc = cuda.cuInit(0)
print("cuInit ->", rc)
dev = ctypes.c_int()
cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
rc = cuda.cuDeviceGet(ctypes.byref(dev), 0)
print("cuDeviceGet ->", rc, "device =", dev.value)

# CUDA context（CUPTI 需要）
ctx = ctypes.c_void_p()
cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
rc = cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)
print("cuCtxCreate ->", rc)
if rc != 0:
    print("警告：context 创建失败，仍尝试枚举")

# ---- CUPTI 回调 ----
CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p,
                      ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)

collected = []

def _cb(name, desc, mid, mtype, userdata):
    try:
        collected.append((name.decode("utf-8", "replace"),
                          desc.decode("utf-8", "replace") if desc else "",
                          mid, mtype))
    except Exception:
        pass

cb = CB(_cb)

cupti.cuptiDeviceEnumMetrics.argtypes = [ctypes.c_int, CB, ctypes.c_void_p]
cupti.cuptiDeviceEnumMetrics.restype = ctypes.c_int

print("\n枚举 metrics ...")
rc = cupti.cuptiDeviceEnumMetrics(dev.value, cb, None)
print("cuptiDeviceEnumMetrics -> rc =", rc)
print("共收集到 %d 个 metric\n" % len(collected))

if not collected:
    print("未取得任何 metric。可能原因：")
    print("  1. CUPTI 需要 profiler 权限（GeForce 默认限制非管理员）")
    print("  2. 该 CUPTI 版本不支持此 GPU 的 metric 枚举")
    sys.exit(0)

# ---- 分类统计 ----
names = [c[0] for c in collected]
tensor = [c for c in collected if "tensor" in c[0].lower() or "hmma" in c[0].lower()]
mma    = [c for c in collected if "mma" in c[0].lower()]
pipe   = [c for c in collected if "pipe" in c[0].lower()]

print("=" * 70)
print("Tensor 相关计数器：%d 个" % len(tensor))
print("=" * 70)
for n, d, mid, mt in tensor[:40]:
    print("  %-58s id=%-5d type=%d" % (n, mid, mt))
if not tensor:
    print("  （无）")

print("\n" + "=" * 70)
print("pipe 类计数器：%d 个（前 20）" % len(pipe))
print("=" * 70)
for n, d, mid, mt in pipe[:20]:
    print("  %-58s id=%d" % (n, mid))

print("\n" + "=" * 70)
print("总览：共 %d 个 metric，其中 tensor=%d, mma=%d, pipe=%d"
      % (len(names), len(tensor), len(mma), len(pipe)))
print("=" * 70)

# 存全量清单
out = "/tmp/cupti_metrics.txt"
with open(out, "w") as f:
    for n, d, mid, mt in sorted(collected):
        f.write("%s\t%d\t%d\t%s\n" % (n, mid, mt, d[:120]))
print("\n全量清单已写入 %s" % out)
