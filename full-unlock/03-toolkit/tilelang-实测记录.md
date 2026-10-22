# TileLang 真实试用记录

**简体中文** | [繁體中文](tilelang-实测记录.zh-TW.md) | [English](tilelang-实测记录.en.md)

> 用户要求「必须真跑」。本文是 TileLang 在 MX450 上的完整实测过程，含三次失败与最终成功。

---

## 一、为什么试它

**搜索结果明确写着：**

> **"tile-lang supports CUDA devices with architectures including Ampere (sm_80+),
> Turing (sm_75), and Volta (sm_70)."**

**这是唯一一个**明确声明支持 sm_75** 的高性能 GEMM DSL。
如果能跑，就是对比本机手写 kernel 的最有力参照。**

---

## 二、四次尝试，三次失败

### 失败 1 · Target 检测失败

```
pip install tilelang          # 成功，0.1.15
```

```python
kernel = tilelang.compile(func, out_idx=[2])
```

```
FAIL: No registered target detector found an available target.
```

**原因：TVM 没有自动识别 CUDA 后端。**

**修复：显式指定 `target="cuda"`。**

### 失败 2 · 缺 CUDA_HOME

```
FAIL: Failed to automatically detect CUDA installation.
      Please set the CUDA_HOME environment variable manually.
```

**检查发现：pip 的 `nvidia-cuda-nvcc-cu12` 包只有 `ptxas`，没有 `nvcc`。**

```
nvidia/cuda_nvcc/bin/ptxas      ← 只有这个
nvidia/cuda_nvcc/include/...    ← 只有头文件
```

**这是 pip 包的固有缺陷：它不提供完整 nvcc 编译器。**

**修复：从 NVIDIA 官方 apt 源装独立组件（不是整个 toolkit）。**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-nvcc-12-9      # 只装编译器，约 27 MB
```

**结果：`/usr/local/cuda-12.9/bin/nvcc`（27 MB），版本 12.9.86。**

**注意：`cuda-nvcc-12-1` 不存在，可用的是 12-5 / 12-6 / 12-8 / 12-9 / 13-0～13-3。**

### 失败 3 · DLPack allocator

```
FAIL: TVM-FFI callee-allocated outputs require Torch's DLPack allocator exchange API.
      Install a compatible torch-c-dlpack-ext or use a supported PyTorch build.
```

**原因：`out_idx=[2]` 让 TileLang 自己分配输出张量，需要 `torch-c-dlpack-ext`。**

**修复（比重装包更快）：改用 caller-allocated 输出。**

```python
# 原来
kernel = tilelang.compile(func, out_idx=[2], target="cuda")
c = kernel(a, b)

# 改成
kernel = tilelang.compile(func, target="cuda")
c = torch.empty(M, N, device="cuda", dtype=torch.float16)
kernel(a, b, c)
```

### 成功 · 正确性 + 性能

```
正确性 (512³):
  max abs err = 0.06250  (相对 5.56e-04)   PASS

性能 (2048³, float16, 强制同步):
  (128,128,32)  编译 1.5s   0.527 TFLOPS  (32.6 ms)
  (64,128,32)   编译 1.4s   0.538 TFLOPS  (31.9 ms)   ← 最佳
  (128,128,64)  FAIL  共享内存 98304 > 上限
  (128,256,32)  FAIL  共享内存 73728 > 上限
```

---

## 三、一个硬件约束被暴露出来

**两个大 tile 配置失败，原因是共享内存超限：**

```
Failed to set the allowed dynamic shared memory size to 98304   (96 KB)
Failed to set the allowed dynamic shared memory size to 73728   (72 KB)
```

**MX450（Turing）的 SM 共享内存上限是 64 KB（需 opt-in 才到 96 KB）。**

**TileLang 没有为 sm_75 做这个 opt-in，所以 block_K=64 这类配置直接失败。**

**这意味着：在这块卡上，TileLang 能用的 tile 配置被显著压缩。**

---

## 四、完整横向对比（全部强制同步）

| 实现 | TFLOPS | 相对手写 | 走的路径 |
|---|---|---|---|
| cuBLAS `TENSOR_OP` | 0.320 | 0.20x | HMMA（模拟） |
| tinygrad（同步后） | 0.347 | 0.22x | HMMA（推测） |
| **TileLang（最佳）** | **0.538** | **0.34x** | **HMMA（cute/CUTLASS）** |
| **本机手写 CUDA C（FFMA）** | **1.603** | **1.00x** | **FFMA** |
| cuBLAS FP32 | 2.122 | 1.32x | FFMA |

**关键观察：**

**所有走 HMMA 的实现（cuBLAS TENSOR_OP / tinygrad / TileLang）都在 0.32–0.54 区间。**

**所有走 FFMA 的实现（手写 / cuBLAS FP32）都在 1.60–2.12 区间。**

**这不是巧合，是 TU117 没有真 TC 的直接后果。**

---

## 五、结论

### TileLang 能用，但不适合这块卡

| 方面 | 评价 |
|---|---|
| sm_75 支持 | ✅ 官方声明支持，实测能编译运行 |
| 正确性 | ✅ PASS（相对误差 5.56e-04，FP16 合理） |
| 性能 | ❌ 0.538，是手写的 34% |
| 大 tile | ❌ 共享内存 64 KB 限制，多个配置失败 |
| 安装门槛 | ⚠️ 需要完整 nvcc（pip 包不够） |

**根本原因：`T.gemm` 在 NVIDIA 上 dispatch 到 cute/CUTLASS，
而 CUTLASS 在 sm_75 上默认走 HMMA。TU117 的 HMMA 是模拟的，所以慢。**

**如果 TileLang 提供「禁用 tensor core」的选项（类似 `tl.dot` 的 input_precision），
理论上能走 FFMA。但当前版本没有暴露这个开关。**

### 这三次失败 + 一次成功，共同证明了一件事

**在 TU117 上，「用高层 DSL」不是最优解。
唯一能拿到 1.6 TFLOPS 的方式，是自己写 CUDA C 并强制走 FFMA。**

**本项目的 `sw_tensor_core_v2.py` 就是这个结论的产物。**

---

## 六、环境记录

```
tilelang    0.1.15
torch       2.3.1+cu121
nvcc        12.9.86（从 NVIDIA apt 源单独安装）
CUDA_HOME   /usr/local/cuda-12.9
GPU         MX450, 14 SM, CC 7.5
共享内存上限 64 KB（96 KB 需 opt-in）
```

**复现：**
```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
python test_tilelang.py
```

**文件：`full-unlock/03-toolkit/test_tilelang.py`**
