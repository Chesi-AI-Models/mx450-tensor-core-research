# -*- coding: utf-8 -*-
"""T2-2 实测: 双缓冲 + 异步预取能否隐藏 T2 传输延迟

背景（MEMORY.md L84 待验证方向）:
  "双缓冲 + 异步预取（cuMemPrefetchAsync）"

T2-3 已证伪"避免回写"（带宽已是 PCIe 3.0 x4 的 80%/83%，无可优化空间）。
所以 T2 只剩两条有效路径：
  T2-1 减少传输「次数」（权重驻留）
  T2-2 隐藏传输「延迟」（双缓冲流水线）  <- 本脚本

实验设计:
  场景 = 权重分块流式加载 + 每块做一次 GEMM（典型的层间推理 / LoRA 训练）
  单缓冲: copy(块i) -> 等 -> GEMM(i) -> copy(块i+1) -> ...        完全串行
  双缓冲: copy(块i+1) 与 GEMM(i) 并行（两个 stream + event）      流水线

  若 GEMM 时间 >= 拷贝时间，双缓冲可把拷贝完全隐藏 -> 总时间 ≈ 纯计算时间
  理论加速上限 = (copy + compute) / max(copy, compute)
"""
import time, torch

print("=" * 78)
print("T2-2: 双缓冲 + 异步预取 实测")
print("=" * 78)
torch.cuda.init()
dev = "cuda"
p = torch.cuda.get_device_properties(0)
print("  GPU:", p.name, "CC=%d.%d" % (p.major, p.minor))
print("  torch:", torch.__version__)

# ---- 参数选择: 让 copy 与 compute 时间接近
M = N_GEMM = 4096
K = 1024
CHUNK_MB = 64
N_CHUNKS = 8

# host 侧 pinned 内存（模拟 GreenBoost 的 T2 pinned 层）
host_blocks = [torch.randn(M, K, pin_memory=True, dtype=torch.float32)
               for _ in range(N_CHUNKS)]
W = torch.randn(K, N_GEMM, device=dev, dtype=torch.float32)
print("  每块权重: %d x %d = %.1f MB" % (M, K, M * K * 4 / 1024 / 1024))
print("  GEMM: %d x %d x %d  (每块 %.1f GFLOP)"
      % (M, N_GEMM, K, 2 * M * N_GEMM * K / 1e9))
print("  块数: %d" % N_CHUNKS)

# ---- 先测纯拷贝与纯计算时间（用于判断是否具备重叠条件）
def bench(fn, iters=5, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

one_dev = torch.empty(M, K, device=dev, dtype=torch.float32)
t_copy = bench(lambda: one_dev.copy_(host_blocks[0], non_blocking=True))
t_gemm = bench(lambda: torch.mm(one_dev, W))
print()
print("  纯拷贝一块      : %7.2f ms  (%.2f GB/s)"
      % (t_copy * 1e3, M * K * 4 / t_copy / 1e9))
print("  纯 GEMM 一块    : %7.2f ms  (%.3f TFLOPS)"
      % (t_gemm * 1e3, 2 * M * N_GEMM * K / t_gemm / 1e12))
print("  重叠条件 copy<=compute ? %s" % ("是 (可完全隐藏)" if t_copy <= t_gemm else "否 (只能部分隐藏)"))
pred_speedup = (t_copy + t_gemm) / max(t_copy, t_gemm)
print("  理论上限加速 = (copy+gemm)/max = %.2fx" % pred_speedup)

# ---- 单缓冲（完全串行）
buffers1 = [torch.empty(M, K, device=dev, dtype=torch.float32)]
acc = torch.zeros(M, N_GEMM, device=dev, dtype=torch.float32)

def run_single():
    out = None
    for i in range(N_CHUNKS):
        buffers1[0].copy_(host_blocks[i], non_blocking=True)
        torch.cuda.synchronize()           # 必须等拷贝完成才能算
        out = torch.mm(buffers1[0], W)
    return out

t0 = time.perf_counter()
run_single()
torch.cuda.synchronize()
t_single = time.perf_counter() - t0
print()
print("-" * 78)
print("  单缓冲串行: %8.2f ms" % (t_single * 1e3))

# ---- 双缓冲 + 双 stream 流水线
bufs = [torch.empty(M, K, device=dev, dtype=torch.float32) for _ in range(2)]
s_copy = torch.cuda.Stream()
s_comp = torch.cuda.Stream()

def run_double():
    out = None
    # 预取第 0 块
    with torch.cuda.stream(s_copy):
        bufs[0].copy_(host_blocks[0], non_blocking=True)
    ev_cur = torch.cuda.Event()
    ev_cur.record(s_copy)

    for i in range(N_CHUNKS):
        # 预取下一块（与本次计算重叠）
        if i + 1 < N_CHUNKS:
            nxt = (i + 1) % 2
            with torch.cuda.stream(s_copy):
                bufs[nxt].copy_(host_blocks[i + 1], non_blocking=True)
            ev_next = torch.cuda.Event()
            ev_next.record(s_copy)
        else:
            ev_next = None
        # 等当前块就绪，然后计算
        s_comp.wait_event(ev_cur)
        with torch.cuda.stream(s_comp):
            out = torch.mm(bufs[i % 2], W)
        ev_cur = ev_next
    torch.cuda.synchronize()
    return out

t0 = time.perf_counter()
out_d = run_double()
t_double = time.perf_counter() - t0
print("  双缓冲流水: %8.2f ms" % (t_double * 1e3))

# ---- 结果
print()
print("=" * 78)
print("结论")
print("=" * 78)
speedup = t_single / t_double
print("  单缓冲 %.2f ms  ->  双缓冲 %.2f ms" % (t_single * 1e3, t_double * 1e3))
print("  实测加速: %.2fx   (理论上限 %.2fx)" % (speedup, pred_speedup))
print("  每块有效耗时: %.2f ms -> %.2f ms"
      % (t_single / N_CHUNKS * 1e3, t_double / N_CHUNKS * 1e3))

# 正确性抽查
try:
    ref = torch.mm(host_blocks[N_CHUNKS - 1].cuda(), W)
    err = (out_d - ref).abs().max().item()
    print("  最后一块结果正确性: max_abs_err = %.4e  %s"
          % (err, "OK" if err < 1.0 else "ERR"))
except Exception as e:
    print("  正确性检查失败:", str(e)[:100])

# 清理
del host_blocks, bufs, buffers1, W, one_dev, acc
torch.cuda.empty_cache()
