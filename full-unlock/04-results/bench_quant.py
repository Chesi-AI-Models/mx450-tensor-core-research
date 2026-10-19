import torch
import time
import os

p = torch.cuda.get_device_properties(0)
print("VRAM: %.2f GiB  headroom=%s"
      % (p.total_memory/1024**3,
         os.environ.get("GREENBOOST_VRAM_HEADROOM_MB", "default")))

def bench_dtype(gb, dtype, iters=5):
    """测不同 dtype 下的有效元素吞吐"""
    esize = torch.tensor([], dtype=dtype).element_size()
    n = int(gb * 1024**3) // esize
    try:
        x = torch.empty(n, dtype=dtype, device="cuda")
        y = torch.empty_like(x)
        x.fill_(1)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            torch.add(x, x, out=y)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        bw = 2 * n * esize / dt / 1e9
        # 有效元素吞吐（每秒处理多少个"逻辑元素"）
        elems = 2 * n / dt / 1e9
        return bw, elems, dt
    except Exception as e:
        return None, None, str(e)[:50]

print()
print("=== 同一 2 GiB 数据量，不同 dtype（走 T2）===")
print("  %-10s %8s %10s %14s %10s" % ("dtype", "元素数", "带宽GB/s", "元素吞吐G/s", "耗时ms"))
print("  " + "-" * 60)

for dt_name, dt in [("float32", torch.float32),
                    ("float16", torch.float16),
                    ("int8", torch.int8)]:
    bw, el, t = bench_dtype(2.0, dt)
    if bw:
        print("  %-10s %8.2fG %10.2f %14.2f %10.1f"
              % (dt_name, 2.0*1024**3/(2 if dt_name=="float16" else (1 if dt_name=="int8" else 4))/1e9, bw, el, t*1000))
    else:
        print("  %-10s FAIL %s" % (dt_name, t))
    torch.cuda.empty_cache()

print()
print("=== 结论 ===")
print("  若带宽(GB/s)在各 dtype 下相同 -> 瓶颈是字节传输，量化收益线性")
print("  若元素吞吐(G/s)相同      -> 瓶颈是元素数，量化无收益")
