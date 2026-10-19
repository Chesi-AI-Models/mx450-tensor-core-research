import time
import numpy as np
from tinygrad import Tensor, Device

print("=== tinygrad 正确性 + 性能验证 ===")
print("Device:", Device.DEFAULT)
print()

# 小尺寸正确性验证
print("[1] 正确性（512x512，对比 numpy）")
N = 512
np.random.seed(42)
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
ref = A @ B

a = Tensor(A)
b = Tensor(B)
c = (a @ b).numpy()
err = np.abs(c - ref).max()
rel = err / np.abs(ref).max()
print("    max abs err = %.5f  (相对 %.2e)" % (err, rel))
print("    判定:", "PASS" if rel < 1e-4 else "FAIL/可疑")

# 大尺寸性能
print()
print("[2] 性能（2048³，浮点操作数）")
N = 2048
for dt in ["float32", "float16"]:
    a = Tensor.rand(N, N, dtype=dt)
    b = Tensor.rand(N, N, dtype=dt)
    (a @ b).realize()
    iters = 10
    t0 = time.time()
    for _ in range(iters):
        c = (a @ b).realize()
    dt_s = (time.time() - t0) / iters
    tf = 2.0 * N**3 / dt_s / 1e12
    print("    %-8s %7.3f TFLOPS  (%.2f ms)" % (dt, tf, dt_s*1000))
    del a, b, c

print()
print("[3] 同一矩阵重复 vs 不同矩阵（检测是否缓存了结果）")
N = 2048
a = Tensor.rand(N, N, dtype="float32").realize()
b = Tensor.rand(N, N, dtype="float32").realize()
c1 = (a @ b).realize()
t0 = time.time()
for _ in range(5):
    c1 = (a @ b).realize()
print("    同矩阵重放: %.2f ms" % ((time.time()-t0)/5*1000))

a2 = Tensor.rand(N, N, dtype="float32").realize()
b2 = Tensor.rand(N, N, dtype="float32").realize()
(a2 @ b2).realize()
t0 = time.time()
for _ in range(5):
    c2 = (a2 @ b2).realize()
print("    换矩阵:     %.2f ms" % ((time.time()-t0)/5*1000))
