#!/usr/bin/env python3
"""
profile_kernel.py - 抓 cuBLAS FP16 matmul 实际执行的 CUDA kernel 名

判读：
  - kernel 名含 "h1688" / "h16816" / "hmma"  -> 走了 Tensor Core 指令
  - kernel 名含 "sgemm" / "gemmSN" / "fp16"  -> CUDA core 路径
  - 若 TC kernel 存在但极慢 -> TC 硬件存在但被限速/数量极少
  - 若无 TC kernel -> 该卡无 TC，cuBLAS 的 TENSOR_OP 算法名只是启发式选择
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = False

SIZE = 2048
print("=" * 70)
print("抓取 FP16 matmul 的 CUDA kernel 名")
print("=" * 70)
print("GPU:", torch.cuda.get_device_name(0))
_p = torch.cuda.get_device_properties(0)
print("capability: %d.%d" % (_p.major, _p.minor))
print()

for dtype, label in [(torch.float16, "FP16"), (torch.float32, "FP32")]:
    a = torch.randn(SIZE, SIZE, device="cuda", dtype=dtype)
    b = torch.randn(SIZE, SIZE, device="cuda", dtype=dtype)
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(5):
            c = a @ b
        torch.cuda.synchronize()

    print("-" * 70)
    print("%s matmul (%dx%d) kernel 列表：" % (label, SIZE, SIZE))
    print("-" * 70)
    evts = prof.key_averages()
    shown = 0
    for e in sorted(evts, key=lambda x: -x.self_device_time_total):
        if e.self_device_time_total <= 0:
            continue
        name = e.key
        # 只显示 gemm/matmul 相关
        if any(k in name.lower() for k in ("gemm", "matmul", "cutlass", "mma")):
            print("  %-72s  %8.2f us  x%d" % (name[:72], e.self_device_time_total, e.count))
            shown += 1
        if shown >= 8:
            break
    if shown == 0:
        print("  （未匹配到 gemm 关键字，打印耗时前 5 项）")
        for e in sorted(evts, key=lambda x: -x.self_device_time_total)[:5]:
            if e.self_device_time_total > 0:
                print("  %-72s  %8.2f us" % (e.key[:72], e.self_device_time_total))
    print()

print("=" * 70)
print("判读：")
print("  h1688 / h16816 / hmma  -> Tensor Core 指令")
print("  sgemm / gemmSN / 其它   -> CUDA core 路径")
print("=" * 70)
