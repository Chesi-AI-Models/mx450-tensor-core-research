import time
import torch
import tilelang
import tilelang.language as T

print("=== TileLang GEMM 测试 ===")
print("tilelang:", tilelang.__version__)
print("torch:", torch.__version__)
p = torch.cuda.get_device_properties(0)
print("GPU: %s  SM=%d  CC=%d.%d" % (p.name, p.multi_processor_count, p.major, p.minor))
print()


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# ---- 正确性（512）----
print("[1] 正确性 512³")
M = N = K = 512
func = matmul(M, N, K, 128, 128, 32)
try:
    kernel = tilelang.compile(func, out_idx=[2])
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).half()
    err = (c.float() - ref.float()).abs().max().item()
    rel = err / ref.float().abs().max().item()
    print("    max abs err = %.5f  (相对 %.2e)  %s" % (err, rel, "PASS" if rel < 1e-3 else "检查"))
except Exception as e:
    print("    FAIL:", str(e)[:200])

print()

# ---- 性能（2048）----
print("[2] 性能 2048³")
M = N = K = 2048
configs = [
    (128, 128, 32),
    (128, 128, 64),
    (64, 128, 32),
    (128, 256, 32),
]

best = None
for bm, bn, bk in configs:
    try:
        func = matmul(M, N, K, bm, bn, bk)
        t0 = time.time()
        kernel = tilelang.compile(func, out_idx=[2])
        ct = time.time() - t0

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        # warmup
        for _ in range(2):
            c = kernel(a, b)
        torch.cuda.synchronize()

        iters = 10
        t0 = time.time()
        for _ in range(iters):
            c = kernel(a, b)
        torch.cuda.synchronize()          # ← 关键：强制同步
        dt = (time.time() - t0) / iters
        tf = 2.0 * M * N * K / dt / 1e12
        print("    (%3d,%3d,%2d)  编译%5.1fs  %7.3f TFLOPS  (%.1f ms)"
              % (bm, bn, bk, ct, tf, dt * 1000))
        if best is None or tf > best[1]:
            best = ((bm, bn, bk), tf)
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        print("    (%3d,%3d,%2d)  FAIL %s" % (bm, bn, bk, str(e).replace("\n", " ")[:80]))

print()
print("=== 对比（全部强制同步）===")
print("  cuBLAS TENSOR_OP : 0.320")
print("  tinygrad         : 0.347")
print("  本机手写 CUDA C  : 1.603")
print("  cuBLAS FP32      : 2.122")
if best:
    print("  TileLang 最佳    : %.3f  (%s)" % (best[1], best[0]))
    print("  vs 手写: %.2fx" % (best[1] / 1.603))
