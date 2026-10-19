import os
import time

from tinygrad import Tensor, Device

print("=== tinygrad GEMM 测试 ===")
print("Device:", Device.DEFAULT)
print("Python:", __import__("sys").version.split()[0])

N = 2048
print("矩阵 %dx%dx%d" % (N, N, N))
print()

results = {}

for dt in ["float32", "float16"]:
    try:
        a = Tensor.rand(N, N, dtype=dt)
        b = Tensor.rand(N, N, dtype=dt)
        # 首次编译
        t0 = time.time()
        c = (a @ b).realize()
        compile_time = time.time() - t0
        print("%s: 首次编译 %.1f s" % (dt, compile_time))

        # 测速
        iters = 5
        t0 = time.time()
        for _ in range(iters):
            c = (a @ b).realize()
        dt_ = (time.time() - t0) / iters
        tf = 2.0 * N**3 / dt_ / 1e12
        print("  %.3f TFLOPS  (%.1f ms/次)" % (tf, dt_ * 1000))
        results[dt] = tf
        del a, b, c
    except Exception as e:
        print("%s: FAIL %s" % (dt, str(e)[:100]))
    print()

print("=== 对比 ===")
print("  本机手写 CUDA C (FFMA)  : 1.603 TFLOPS")
print("  cuBLAS TENSOR_OP        : 0.320 TFLOPS")
for k, v in results.items():
    print("  tinygrad %-8s       : %.3f TFLOPS  (vs 手写 %.2fx)" % (k, v, v / 1.603))
