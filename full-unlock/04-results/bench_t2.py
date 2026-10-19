import torch
import time
import os

p = torch.cuda.get_device_properties(0)
print("VRAM: %.2f GiB" % (p.total_memory / 1024**3))
print("GREENBOOST_USE_DMA_BUF =", os.environ.get("GREENBOOST_USE_DMA_BUF", "(unset)"))
print()

def bench_matmul(size, dtype=torch.float16, iters=10, label=""):
    try:
        a = torch.randn(size, size, device="cuda", dtype=dtype)
        b = torch.randn(size, size, device="cuda", dtype=dtype)
        for _ in range(2):
            c = a @ b
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            c = a @ b
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        tf = 2.0 * size**3 / dt / 1e12
        mem = (a.numel() + b.numel() + c.numel()) * 2 / 1024**3
        print("  %-28s size=%4d  mem=%5.2f GiB  %7.3f TFLOPS  %7.1f ms"
              % (label, size, mem, tf, dt*1000))
        del a, b, c
        torch.cuda.empty_cache()
        return tf
    except Exception as e:
        print("  %-28s size=%4d  FAIL %s" % (label, size, str(e)[:60]))
        return None

print("=== 基准：纯 VRAM 内的 matmul ===")
t_vram = bench_matmul(2048, label="2048 (fits in VRAM)")

print()
print("=== 超过 VRAM 的 matmul（走 T2）===")
# 4096x4096 fp16 = 32MB per matrix，3 个 = 96MB，还在 VRAM 内
# 要超过 VRAM，用更大的 size
# 6144x6144 fp16 = 72MB per matrix, 3 个 = 216MB
# 但真正压到 T2 需要 >1.6GB
# 用 16384x16384 = 512MB per matrix, 3 个 = 1.5GB，接近上限
for s in [4096, 8192, 12288, 16384]:
    bench_matmul(s, label="%d (pressure)" % s)

print()
print("=== 显存统计 ===")
print("allocated: %.2f GiB" % (torch.cuda.memory_allocated()/1024**3))
print("reserved : %.2f GiB" % (torch.cuda.memory_reserved()/1024**3))
print("max alloc: %.2f GiB" % (torch.cuda.max_memory_allocated()/1024**3))
