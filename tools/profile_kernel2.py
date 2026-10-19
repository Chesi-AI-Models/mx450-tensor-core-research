#!/usr/bin/env python3
"""profile_kernel2.py - 抓 FP16/FP32 matmul 的 CUDA kernel 名（兼容多版本 API）"""
import torch

torch.backends.cuda.matmul.allow_tf32 = False
SIZE = 2048

print("=" * 72)
print("FP16 / FP32 matmul 实际执行的 CUDA kernel")
print("=" * 72)
p = torch.cuda.get_device_properties(0)
print("GPU: %s   SM=%d   capability=%d.%d" % (p.name, p.multi_processor_count, p.major, p.minor))
print()

for dtype, label in [(torch.float16, "FP16"), (torch.float32, "FP32")]:
    a = torch.randn(SIZE, SIZE, device="cuda", dtype=dtype)
    b = torch.randn(SIZE, SIZE, device="cuda", dtype=dtype)
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            c = a @ b
        torch.cuda.synchronize()

    print("-" * 72)
    print("%s matmul (%dx%d)" % (label, SIZE, SIZE))
    print("-" * 72)
    evts = prof.key_averages()
    rows = []
    for e in evts:
        # 兼容不同版本的字段名
        t = getattr(e, "self_device_time_total", None)
        if t is None:
            t = getattr(e, "self_cuda_time_total", 0)
        rows.append((e.key, t, e.count))
    rows.sort(key=lambda r: -r[1])
    n = 0
    for key, t, cnt in rows:
        if t <= 0:
            continue
        mark = ""
        kl = key.lower()
        if any(k in kl for k in ("h1688", "h16816", "hmma", "tensor")):
            mark = "  <== TENSOR CORE"
        elif any(k in kl for k in ("sgemm", "gemmsn", "gemm")):
            mark = "  <== gemm"
        print("  %-64s %9.1f us x%-3d%s" % (key[:64], t, cnt, mark))
        n += 1
        if n >= 10:
            break
    print()

print("=" * 72)
print("判读：kernel 名含 h1688/h16816/hmma -> Tensor Core 指令")
print("      sgemm/gemmSN/其它             -> CUDA core 路径")
print("=" * 72)
