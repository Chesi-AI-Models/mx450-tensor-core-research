import time
from tinygrad import Tensor, Device

print("=== tinygrad 同步验证 ===")
print("Device:", Device.DEFAULT)
print()

N = 2048
a = Tensor.rand(N, N, dtype="float32").realize()
b = Tensor.rand(N, N, dtype="float32").realize()

# 方法 1：只 realize（可能异步）
(a @ b).realize()
t0 = time.time()
for _ in range(5):
    c = (a @ b).realize()
t_realize = (time.time() - t0) / 5

# 方法 2：realize + numpy（强制同步回传）
t0 = time.time()
for _ in range(5):
    c = (a @ b).numpy()
t_numpy = (time.time() - t0) / 5

# 方法 3：用 Device 的 synchronize（如果存在）
try:
    from tinygrad.runtime.ops_cuda import CUDADevice
    dev = Device[Device.DEFAULT]
    sync = getattr(dev, "synchronize", None)
    print("Device.synchronize 存在:", sync is not None)
except Exception as e:
    print("取 synchronize 失败:", str(e)[:60])
    sync = None

print()
print("%-30s %10s %12s" % ("方法", "耗时ms", "TFLOPS"))
print("-" * 56)

for name, t in [("realize()（可能异步）", t_realize),
                ("realize()+numpy()（强制同步）", t_numpy)]:
    print("%-30s %10.2f %12.3f" % (name, t * 1000, 2.0 * N**3 / t / 1e12))

if sync:
    (a @ b).realize()
    sync()
    t0 = time.time()
    for _ in range(5):
        c = (a @ b).realize()
        sync()
    t_sync = (time.time() - t0) / 5
    print("%-30s %10.2f %12.3f" % ("realize()+synchronize()", t_sync*1000, 2.0*N**3/t_sync/1e12))

print()
print("参考：cuBLAS FP32 = 2.122 TFLOPS，PTX FP32 FMA 峰值 = 3.06 TFLOPS")
print("若同步后仍 >3.1，说明存在测量或算法问题")
