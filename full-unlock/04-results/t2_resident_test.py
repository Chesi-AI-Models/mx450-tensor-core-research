# -*- coding: utf-8 -*-
"""T2-1 实测: 权重驻留（热点层常驻 VRAM）能省多少

背景（MEMORY.md L83 待验证方向，T2 最后一项）:
  "权重驻留策略（热点层常驻 VRAM）"

原理:
  T2-2 解决「单次传输的延迟」，T2-1 解决「传输次数」。
  推理时每个 batch / 每个 token 都要过全部层。若某层每次都从 T2 重拷，
  拷贝量 = batch数 × 层数 × 层大小；若驻留 VRAM，只需拷一次。

  无驻留总时间 = B × L × (copy + gemm)
  全驻留总时间 = L × copy + B × L × gemm
  部分驻留(P 层) = P×copy + B×(L-P)×copy + B×L×gemm

  理论加速 = B×L×(cop+gemm) / (L×copy + B×L×gemm)

实验:
  8 层 × 16MB，4 个 batch，对比三种策略，并验证公式
"""
import time, torch

print("=" * 78)
print("T2-1: 权重驻留（热点层常驻 VRAM）实测")
print("=" * 78)
torch.cuda.init()
dev = "cuda"
p = torch.cuda.get_device_properties(0)
print("  GPU:", p.name, "VRAM %.2f GiB" % (p.total_memory / 1024**3))

M, K, N = 4096, 1024, 4096
L = 8            # 层数
BATCH = 4        # 重复跑几个 batch（模拟多 token / 多轮）
P_RESIDENT = 4   # 部分驻留时驻留几层

layer_bytes = M * K * 4
print()
print("  层配置: %d 层 × %.1f MB = %.1f MB" % (L, layer_bytes / 1024**2, L * layer_bytes / 1024**2))
print("  batch 数: %d   每层 GEMM: %d×%d×%d (%.1f GFLOP)"
      % (BATCH, M, N, K, 2 * M * N * K / 1e9))
print("  部分驻留层数: %d" % P_RESIDENT)

# host 侧 pinned（模拟 T2）
host = [torch.randn(M, K, pin_memory=True, dtype=torch.float32) for _ in range(L)]
W = torch.randn(K, N, device=dev, dtype=torch.float32)

# 标定单次 copy / gemm
one = torch.empty(M, K, device=dev, dtype=torch.float32)
def bench(fn, iters=5, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

t_copy = bench(lambda: one.copy_(host[0], non_blocking=True))
t_gemm = bench(lambda: torch.mm(one, W))
print()
print("  单次拷贝: %.2f ms (%.2f GB/s)   单次 GEMM: %.2f ms (%.3f TFLOPS)"
      % (t_copy * 1e3, layer_bytes / t_copy / 1e9, t_gemm * 1e3,
         2 * M * N * K / t_gemm / 1e12))

# ---------- 策略 A: 无驻留（每 batch 每层都重拷）
buf = torch.empty(M, K, device=dev, dtype=torch.float32)
def run_nores():
    for b in range(BATCH):
        for i in range(L):
            buf.copy_(host[i], non_blocking=True)
            torch.cuda.synchronize()
            torch.mm(buf, W)

# ---------- 策略 B: 全驻留
dev_layers = [torch.empty(M, K, device=dev, dtype=torch.float32) for _ in range(L)]
def run_fullres():
    for i in range(L):
        dev_layers[i].copy_(host[i], non_blocking=True)
    torch.cuda.synchronize()
    for b in range(BATCH):
        for i in range(L):
            torch.mm(dev_layers[i], W)

# ---------- 策略 C: 部分驻留（前 P 层驻留，其余流式）
partial = [torch.empty(M, K, device=dev, dtype=torch.float32) for _ in range(P_RESIDENT)]
def run_partial():
    for i in range(P_RESIDENT):
        partial[i].copy_(host[i], non_blocking=True)
    torch.cuda.synchronize()
    for b in range(BATCH):
        for i in range(P_RESIDENT):
            torch.mm(partial[i], W)
        for i in range(P_RESIDENT, L):
            buf.copy_(host[i], non_blocking=True)
            torch.cuda.synchronize()
            torch.mm(buf, W)

def timed(fn, label):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    return t

print()
print("-" * 78)
tA = timed(run_nores, "A")
print("  A 无驻留      : %8.2f ms   (拷贝 %d 次)" % (tA * 1e3, BATCH * L))
tB = timed(run_fullres, "B")
print("  B 全驻留      : %8.2f ms   (拷贝 %d 次)" % (tB * 1e3, L))
tC = timed(run_partial, "C")
print("  C 部分驻留(%d层): %8.2f ms   (拷贝 %d 次)"
      % (P_RESIDENT, tC * 1e3, P_RESIDENT + BATCH * (L - P_RESIDENT)))

print()
print("=" * 78)
print("结论")
print("=" * 78)
print("  B/A = %.2fx    C/A = %.2fx" % (tA / tB, tA / tC))

# 理论公式验证
def predict(ncopy):
    return ncopy * t_copy + BATCH * L * t_gemm
print()
print("  理论公式验证（预测 vs 实测）:")
print("    A 预测 %7.2f ms  实测 %7.2f ms  误差 %+.1f%%"
      % (predict(BATCH * L) * 1e3, tA * 1e3,
         (tA / predict(BATCH * L) - 1) * 100))
print("    B 预测 %7.2f ms  实测 %7.2f ms  误差 %+.1f%%"
      % (predict(L) * 1e3, tB * 1e3, (tB / predict(L) - 1) * 100))
print("    C 预测 %7.2f ms  实测 %7.2f ms  误差 %+.1f%%"
      % (predict(P_RESIDENT + BATCH * (L - P_RESIDENT)) * 1e3, tC * 1e3,
         (tC / predict(P_RESIDENT + BATCH * (L - P_RESIDENT)) - 1) * 100))

# ---------- 外推到 7B 模型
print()
print("=" * 78)
print("外推：7B 模型（32 层，FP16 每层约 448 MB，总 14 GB）")
print("=" * 78)
L7 = 32
TOTAL_MB = 14 * 1024          # 14336 MB  （统一用 MB，避免 MB/GB 混用）
layer7_mb = TOTAL_MB / L7     # 448 MB
VRAM_USABLE_MB = 1.5 * 1024   # 1536 MB，留余量
resident_layers = int(VRAM_USABLE_MB / layer7_mb)
resident_mb = resident_layers * layer7_mb
stream_mb = TOTAL_MB - resident_mb
BW = 3.15                     # GB/s 实测

print("  每层 %.0f MB，可用 VRAM %.0f MB → 最多驻留 %d / %d 层（%.0f MB）"
      % (layer7_mb, VRAM_USABLE_MB, resident_layers, L7, resident_mb))
print()
print("  %-8s %-24s %-24s %s"
      % ("batch", "无驻留传输", "部分驻留传输", "省"))
print("  " + "-" * 74)
for batch in (1, 2, 4, 8, 16, 32):
    # 无驻留：每 batch 都传全部
    t_no_mb = TOTAL_MB * batch
    # 部分驻留：驻留层只传一次，流式层每 batch 传
    t_part_mb = resident_mb + stream_mb * batch
    t_no_s = t_no_mb / 1024 / BW
    t_part_s = t_part_mb / 1024 / BW
    save = (1 - t_part_s / t_no_s) * 100
    print("  %-8d %7.1f GB  %6.2fs      %7.1f GB  %6.2fs      %5.1f%%"
          % (batch, t_no_mb / 1024, t_no_s, t_part_mb / 1024, t_part_s, save))

print()
print("  解读：")
print("   - 驻留层只付一次拷贝成本，之后每个 batch 都省下这部分传输")
print("   - batch 越大，驻留收益越接近上限 = resident_mb / TOTAL_MB = %.0f%%"
      % (resident_mb / TOTAL_MB * 100))
print("   - MX450 只能驻留 %d/32 层（%.0f%% 权重），所以单机上限就是 %.0f%%"
      % (resident_layers, resident_mb / TOTAL_MB * 100, resident_mb / TOTAL_MB * 100))
print("   - 配合 4-bit 量化（14GB→3.5GB）后，可驻留层数变为 %.0f/32"
      % (VRAM_USABLE_MB / (TOTAL_MB / 4 / L7)))
print("     → INT4 下驻留比例可达 %.0f%%，这才是量化 + 驻留的组合价值"
      % (min(1.0, VRAM_USABLE_MB / (TOTAL_MB / 4)) * 100))

del host, dev_layers, partial, W, one, buf
torch.cuda.empty_cache()
