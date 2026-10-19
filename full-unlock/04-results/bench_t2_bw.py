import torch
import time
import os

p = torch.cuda.get_device_properties(0)
print("VRAM: %.2f GiB   DMA_BUF=%s" % (p.total_memory/1024**3,
      os.environ.get("GREENBOOST_USE_DMA_BUF", "unset")))
print()

def bench_copy(gb, iters=5, label=""):
    """测 elementwise 带宽：y = x * 2 + 1，受内存带宽限制"""
    n = int(gb * 1024**3) // 4   # float32
    try:
        x = torch.empty(n, dtype=torch.float32, device="cuda")
        x.fill_(1.0)
        y = torch.empty_like(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            torch.add(x, 1.0, out=y)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        # 读写各 n*4 字节
        bw = 2 * n * 4 / dt / 1e9
        print("  %-22s %5.1f GiB  %7.2f GB/s  %7.1f ms" % (label, gb, bw, dt*1000))
        del x, y
        torch.cuda.empty_cache()
        return bw
    except Exception as e:
        print("  %-22s %5.1f GiB  FAIL %s" % (label, gb, str(e)[:50]))
        return None

print("=== elementwise 带宽测试 ===")
print("  %-22s %5s      %7s      %7s" % ("场景", "大小", "带宽", "耗时"))
print("  " + "-" * 56)

bench_copy(0.5, label="0.5G (in VRAM)")
bench_copy(1.0, label="1.0G (in VRAM)")

print()
print("=== 超过 VRAM（走 T2）===")
for gb in [2, 3, 4]:
    bench_copy(gb, label="%dG (T2)" % gb)

print()
print("=== 分配统计 ===")
print("  max allocated: %.2f GiB" % (torch.cuda.max_memory_allocated()/1024**3))
