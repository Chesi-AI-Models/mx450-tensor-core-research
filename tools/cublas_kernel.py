#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""让 cuBLAS 打印它实际选择的 kernel，并对小矩阵做速度对比"""
import torch, time, os, subprocess

dev = torch.device("cuda")

def speed(n, dtype, iters=50):
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(5): c = a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): c = a @ b
    torch.cuda.synchronize()
    dt = (time.time()-t0)/iters
    tf = 2*n**3/dt/1e12
    del a,b,c; torch.cuda.empty_cache()
    return tf

print("="*66)
print("小矩阵 GEMM（放在 L2/显存内可容纳范围内，排除带宽干扰）")
print("="*66)
for n in (512, 1024, 2048):
    t32 = speed(n, torch.float32)
    t16 = speed(n, torch.float16)
    print("  n=%-5d  FP32 %6.3f TFLOPS   FP16 %6.3f TFLOPS   比值 %.2f×"
          % (n, t32, t16, t16/max(t32,1e-6)))
print()
print("  参考：若 TC 可用，FP16/FP32 比值应为 2~10×；")
print("        若比值 ≈1，说明 FP16 只是跑在 CUDA 核上，TC 未参与。")
print("="*66)

print("\n现在让 cuBLAS 打印它选的 kernel（CUBLAS_LOGINFO_DBG）...\n")
env = dict(os.environ)
env["CUBLAS_LOGDEST_DBG"] = "stdout"
env["CUBLAS_LOGINFO_DBG"] = "1"
code = (
    "import torch;"
    "a=torch.randn(1024,1024,device='cuda',dtype=torch.float16);"
    "b=a@a;torch.cuda.synchronize();"
    "c=torch.randn(1024,1024,device='cuda',dtype=torch.float32);"
    "d=c@c;torch.cuda.synchronize();print('ok')"
)
r = subprocess.run(["/vol1/ganlin/venv/bin/python", "-c", code],
                   capture_output=True, text=True, env=env, timeout=300)
lines = (r.stdout + r.stderr).splitlines()
hits = [l for l in lines if any(k in l.lower() for k in
        ("gemm", "kernel", "h884", "h1688", "tensor", "sgemm", "hgemm", "name"))]
print("---- cuBLAS 日志（筛出 kernel 相关）----")
for l in hits[:40]:
    print("  " + l.strip()[:160])
if not hits:
    print("  （无 kernel 级日志，日志级别可能被裁剪）")
    for l in lines[:15]:
        print("  " + l.strip()[:160])
print("\n返回码:", r.returncode)
