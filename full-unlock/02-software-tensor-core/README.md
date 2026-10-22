# 软件 Tensor Core：在无 TC 的卡上重建矩阵乘

**简体中文** | [繁體中文](README.zh-TW.md) | [English](README.en.md)

> 目标：TU117（MX450）没有 Tensor Core，用 CUDA core 实现等效的矩阵乘加速。
>
> 最终结果：**1.603 TFLOPS**，是 cuBLAS `TENSOR_OP` 路径（0.320）的 **5.0 倍**，
> 且 `max abs err = 0.00000`（与 float64 参考完全一致）。

---

## 一、为什么需要它

| 路径 | 实测 TFLOPS | 说明 |
|---|---|---|
| cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | 0.320 | cuBLAS 认为有 TC，选了 `cutlass_70_tensorop_f16_s884gemm`，但 TU117 没 TC，实际由 FP16 单元模拟 |
| cuBLAS FP32 | 2.122 | 已达理论峰值（2.5）的 85% |
| FP16x2 纯计算峰值 | 5.770 | **FP16 累加**，K=2048 会溢出，不能做真实 GEMM |

**所以：cuBLAS 在 FP16 上给了你一条 0.32 的死路（因为它以为有 TC）。
自己写 FFMA kernel 能拿到 1.6 —— 这就是 5 倍提升的来源。**

---

## 二、四版演进（含两次失败）

| 版本 | 文件 | 方案 | 结果 |
|---|---|---|---|
| v1 | `triton_gemm.py` | Triton `tl.dot`，FP16 输入 | 0.595 TFLOPS（走 HMMA） |
| v2 | `triton_gemm_v2.py` | 外积（`tl.where`+`tl.sum` 避开 dot） | 0.122（生成大量 shuffle，比 HMMA 还慢 5 倍） |
| v4 | `triton_gemm_v4.py` | Triton `tl.dot`，FP32 输入 | **编译崩溃 `map::at`**（Triton 在 CC 7.5 上的已知 bug，triton-lang/triton#5557） |
| **final** | **`sw_tensor_core_v2.py`** | **NVRTC + CUDA C + FFMA** | **1.603 TFLOPS** ✅ |

### 关键教训

**Triton 在 CC 7.5 上不可用于 FP32。** 官方 issue 明确写了：
FP32 原生支持在计算能力 7.5 上有限制，编译器在生成 LLVM IR 时丢失类型元信息，
引发 `map::at` 崩溃。切换到 FP16 可绕过——但 FP16 会走 HMMA，正是我们要避开的。

**所以必须绕开 Triton，直接生成代码。**

---

## 三、最终实现（`sw_tensor_core_v2.py`）

### 技术栈

```
CUDA C 源码（unsigned short + inline PTX）
    ↓ libnvrtc.so.12（NVRTC）
PTX（39,393 字符）
    ↓ cuModuleLoadData（libcuda）
GPU 执行
```

**不用 `cuda_fp16.h`**（NVRTC 默认 include 路径里没有），全部用：

```cuda
__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
```

### Kernel 结构

```
BLOCK: 128 × 128，BK = 16，256 线程
每线程: 8 × 8 = 64 个 float 累加器
共享内存: As[128][16] + Bs[16][128]（half，共 8 KB）

每个 k:
  读 8 个 a（共享内存 → 寄存器）
  读 8 个 b
  64 次 FFMA
  → FMA/加载比 = 64/16 = 4
```

### 已验证的 PTX 特征

```
含 cvt.f32.f16 : True
含 fma.rn.f32  : True
含 mma.sync    : False   ← 关键：完全绕开 HMMA
```

---

## 四、性能数据

**矩阵 2048³，10 次迭代取平均：**

```
1.603 TFLOPS   (10.7 ms/次)
max abs err = 0.00000
```

| 对比对象 | TFLOPS | 倍数 |
|---|---|---|
| cuBLAS TENSOR_OP | 0.320 | **5.0x** |
| cuBLAS FP32 | 2.122 | 0.76x |
| FP16x2 纯计算峰值 | 5.770 | 0.28x（不可比，FP16 累加） |

### 为什么是 1.603 而不是 5.77

**`fma.rn.f16x2` 是 FP16 累加。K=2048 时累加器溢出，结果不可用。**

真实 GEMM 必须 FP32 累加 → 只能 FFMA → 硬件上限是 FP32 的 2.5 TFLOPS。
1.603 / 2.5 = **64% 效率**，剩余损耗来自共享内存带宽和指令调度。

**这是硬件的算术强度上限，不是实现缺陷。**

### 尝试过但更慢的优化

| 优化 | 结果 | 原因 |
|---|---|---|
| 4×4 tile（BM=BN=64） | 1.539 | FMA/加载比只有 2 |
| **8×8 tile（BM=BN=128）** | **1.603** | ✅ 最优 |
| 共享内存改 float | 1.427 | 共享内存翻倍，bank conflict 增加 |

---

## 五、踩过的坑（ctypes 调用 CUDA Driver API）

1. **必须显式声明 `argtypes`**
   否则 64 位指针被截断成 int → `cuModuleLoadData` 处 **SIGSEGV**。
   这是本项目耗时最久的一个 bug。

2. **必须复用 PyTorch 的 CUDA context**
   自己 `cuCtxCreate_v2` 会创建第二个 context，模块加载到错误的 context 里，
   表现为"加载成功但调用无效"。
   正确做法：`torch.cuda.init()` → `cuCtxGetCurrent()`。

3. **`cuModuleLoadData` 直接吃 PTX 文本**
   不需要 ptxas，不需要 nvcc，驱动内部 JIT。

---

## 六、复现

```bash
/vol1/ganlin/venv/bin/python sw_tensor_core_v2.py --size 2048
```

依赖：`torch 2.3.1` + `nvidia-cuda-nvrtc-cu12`（提供 `libnvrtc.so.12`）。

---

## 七、这个结果的意义

**MX450 的矩阵乘真实上限是 2.5 TFLOPS（FP32），cuBLAS 已经做到 2.12。**

**但 cuBLAS 的 FP16 路径只有 0.32——因为它以为有 TC。**

**所以「软件 TC」的真正价值不是超过 FP32，而是：**

**当你的模型是 FP16 时，不要用 cuBLAS 的 FP16 路径（0.32），
用自定义的 half→float→FFMA kernel（1.6），快 5 倍。**

**这就是在无 TC 硬件上能做的最优解。**
