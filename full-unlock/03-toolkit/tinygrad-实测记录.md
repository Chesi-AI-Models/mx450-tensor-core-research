# tinygrad 真实试用记录（含一个重大测量陷阱）

> 用户要求「必须真跑不能从文档判断」。本文是 tinygrad 的实际执行记录。
>
> **核心发现：tinygrad 的 lazy 求值会让性能测量产生 23 倍的虚高。**

---

## 一、安装与首测

```bash
pip install tinygrad      # 成功，走 PyPI（GitHub 直连在 NAS 上不稳定）
```

```python
from tinygrad import Tensor, Device
print(Device.DEFAULT)     # CUDA
```

**后端识别正确（CUDA）。**

---

## 二、第一次测性能：8.0 TFLOPS —— 但这是假的

### 测量代码

```python
N = 2048
a = Tensor.rand(N, N, dtype="float32")
b = Tensor.rand(N, N, dtype="float32")
(a @ b).realize()          # 首次编译

t0 = time.time()
for _ in range(10):
    c = (a @ b).realize()  # ← 问题在这里
dt = (time.time() - t0) / 10
```

### 结果

```
float32:  7.638 TFLOPS   (2.25 ms)
float16:  4.686 TFLOPS   (3.67 ms)
```

### 为什么可疑

| 检查项 | 数据 | 判断 |
|---|---|---|
| MX450 FP32 理论峰值 | 896 CUDA × 2 FMA × 1.6 GHz = **2.87 TFLOPS** | 7.64 是 **2.7 倍** |
| 换算所需频率 | 7.638e12 / (896×2) = **4.26 GHz** | MX450 不可能 |
| cuBLAS FP32 实测 | **2.122 TFLOPS** | tinygrad 不可能快 3.6 倍 |
| PTX FP32 FMA 纯计算峰值 | **3.06 TFLOPS** | 7.64 超过纯计算上限 |

**四个独立参照全部矛盾 → 一定是测量问题。**

---

## 三、定位问题：tinygrad 是 lazy 的

**tinygrad 的设计核心是惰性求值**：算子先记进 UOp 图，只在需要结果时才真正执行。

**`.realize()` 只是「把图变成 kernel 并**排队**」，不保证执行完成。**

**所以计时测到的是「排队 + 启动开销」，不是「核函数运行时间」。**

**矩阵越大，虚高越明显——因为真正的计算被推到计时区之外。**

---

## 四、强制同步后的真实数据

```python
# 方法 1：只 realize（异步）
t0 = time.time()
for _ in range(5):
    c = (a @ b).realize()
t_realize = (time.time() - t0) / 5

# 方法 2：realize + numpy（强制回传，必然同步）
t0 = time.time()
for _ in range(5):
    c = (a @ b).numpy()
t_numpy = (time.time() - t0) / 5

# 方法 3：realize + Device.synchronize()
dev = Device[Device.DEFAULT]
t0 = time.time()
for _ in range(5):
    c = (a @ b).realize()
    dev.synchronize()
t_sync = (time.time() - t0) / 5
```

### 结果（2048³，float32）

| 方法 | 耗时 | TFLOPS | 判定 |
|---|---|---|---|
| `realize()` 异步 | 2.14 ms | **8.029** | ❌ 假的 |
| `realize()+numpy()` | 116.64 ms | **0.147** | ✅ 真实（含 D2H 拷贝） |
| `realize()+synchronize()` | 49.44 ms | **0.347** | ✅ 真实（纯 GPU 时间） |

**真实性能：0.347 TFLOPS。虚高 23 倍。**

---

## 五、修正后的完整对比

| 实现 | TFLOPS | 相对 cuBLAS TENSOR_OP |
|---|---|---|
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | 0.320 | 1.0x |
| **tinygrad（同步后）** | **0.347** | **1.08x** |
| **本机手写 CUDA C（FFMA，NVRTC）** | **1.603** | **5.0x** |
| cuBLAS FP32 | 2.122 | 6.6x |

**结论：tinygrad 的 matmul 与 cuBLAS TENSOR_OP 同级（0.32–0.35），
说明它同样走了 HMMA 路径（TU117 上的模拟 TC）。**

**而手写 FFMA kernel 快 4.6 倍。**

**这反过来验证了本项目的核心成果：在无 TC 的卡上，绕开 HMMA 走 FFMA 是正确策略。**

---

## 六、正确性验证

```python
N = 512
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
ref = A @ B

c = (Tensor(A) @ Tensor(B)).numpy()
err = np.abs(c - ref).max()      # 0.00007
rel = err / np.abs(ref).max()    # 6.91e-07
```

**判定：PASS（相对误差 6.9e-07，符合 FP32 预期）。**

**说明：tinygrad 的计算是正确的，只是计时方法有陷阱。**

---

## 七、这个陷阱的普遍性

**任何用 tinygrad 做性能对比的结论，都必须确认是否加了同步。**

**搜索到的所有 tinygrad 教程/评测，都只演示 `DEBUG=4` 看 kernel，没有提醒这一点。**

**这不是 tinygrad 的 bug（lazy 是它的设计），而是使用者容易踩的坑。**

**本项目记录在此，供后续参考。**

---

## 八、对本项目的实际结论

**tinygrad 对 MX450 的价值：**

| 方面 | 评价 |
|---|---|
| 性能 | ❌ 不如手写 FFMA（0.347 vs 1.603） |
| 正确性 | ✅ 通过 |
| 可调试性 | ✅ 优秀（`DEBUG=4` 直接看生成的 kernel） |
| 对 TU117 的适配 | ⚠️ 走 HMMA，没有针对无 TC 优化 |
| 作为参考实现 | ✅ 值得读它的 CUDA 后端代码 |

**结论：tinygrad 不能直接用来加速 MX450，但它的 `DEBUG=4` 可以看 kernel 生成，
对手写优化有参考价值。**

---

## 九、复现

```bash
pip install tinygrad
python verify_tinygrad.py           # 正确性 + 性能（含异步陷阱复现）
python verify_tinygrad_sync.py      # 三种同步方式的对比
```

**文件位置：**
- `full-unlock/03-toolkit/test_tinygrad.py`（首测）
- `full-unlock/03-toolkit/verify_tinygrad.py`（正确性验证）
- `full-unlock/03-toolkit/verify_tinygrad_sync.py`（同步对比）
