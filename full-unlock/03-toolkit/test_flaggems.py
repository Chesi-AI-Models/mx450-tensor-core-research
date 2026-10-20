#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FlagGems 在 MX450 (TU117 / CC 7.5) 上的实测

目的:
  1. 验证 FlagGems 能否在无 Tensor Core 的 TU117 上真正跑起来
  2. 量化 Triton 实现的 mm 相对 cuBLAS 的差距
  3. 弄清 "Unsupported GPU arch specialization" 到底是不是致命错误
  4. 统计可用的算子数量

每一步独立 try, 任何一个崩掉不影响后续。
"""
import sys, time, traceback
import torch

def sec(t):
    print()
    print("-" * 72)
    print(t)
    print("-" * 72)

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            raise
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

print("=" * 72)
print("FlagGems on MX450 (TU117, CC 7.5, no Tensor Core)")
print("=" * 72)
print("python :", sys.version.split()[0])
print("torch  :", torch.__version__)
p = torch.cuda.get_device_properties(0)
print("GPU    : %s  CC=%d.%d  SM=%d  VRAM=%.2f GiB"
      % (p.name, p.major, p.minor, p.multi_processor_count,
         p.total_memory / 1024**3))
try:
    import triton
    print("triton :", triton.__version__)
except Exception as e:
    print("triton : NOT AVAILABLE", e)

# ---------------------------------------------------------------- baseline
sec("[1] baseline: torch.mm (cuBLAS)")
N = 1024
a16 = torch.randn(N, N, device="cuda", dtype=torch.float16)
b16 = torch.randn(N, N, device="cuda", dtype=torch.float16)
a32 = a16.float(); b32 = b16.float()

res = {}
for tag, x, y in (("fp16", a16, b16), ("fp32", a32, b32)):
    try:
        t = bench(lambda: torch.mm(x, y))
        tf = 2.0 * N**3 / t / 1e12
        res["cublas_" + tag] = tf
        print("    cuBLAS %s  %7.3f ms  ->  %7.4f TFLOPS" % (tag, t * 1e3, tf))
    except Exception as e:
        print("    cuBLAS %s  FAIL: %s" % (tag, str(e)[:120]))

# ---------------------------------------------------------------- import
sec("[2] import flag_gems")
try:
    import flag_gems
    print("    version :", getattr(flag_gems, "__version__", "?"))
    print("    device  :", getattr(flag_gems, "device", "?"))
    print("    exported symbols:", len([x for x in dir(flag_gems) if not x.startswith("_")]))
except Exception:
    traceback.print_exc()
    raise SystemExit(1)

# ---------------------------------------------------------------- arch state
sec("[3] 后端/架构特化状态 (关键: Unsupported 是否致命)")
try:
    from flag_gems.runtime import backend as _be
    st = _be._state
    for attr in ("vendor_name", "device_name", "has_arch", "arch", "vendor_module"):
        if hasattr(st, attr):
            v = getattr(st, attr)
            if hasattr(v, "__name__"):
                v = v.__name__
            print("    _state.%-14s = %s" % (attr, v))
    # 显式打印 ARCH_MAP 支持哪些
    try:
        vm = st.vendor_module
        am = getattr(vm, "ARCH_MAP", None)
        print("    vendor ARCH_MAP keys =", list(am.keys()) if am else am)
    except Exception as e:
        print("    ARCH_MAP 读取失败:", str(e)[:100])
except Exception:
    traceback.print_exc()

# ---------------------------------------------------------------- mm test
sec("[4] use_gems() 下的 mm")
try:
    with flag_gems.use_gems():
        c_gems = torch.mm(a16, b16)
        torch.cuda.synchronize()
    ref = torch.mm(a16, b16)
    err = (c_gems.float() - ref.float()).abs().max().item()
    rel = err / ref.float().abs().max().item()
    print("    correctness: max_abs_err=%.5f  rel=%.5f" % (err, rel))

    # 计时: 显式调 flag_gems.mm 避免上下文开销
    if hasattr(flag_gems, "mm"):
        t = bench(lambda: flag_gems.mm(a16, b16))
        tf = 2.0 * N**3 / t / 1e12
        res["flaggems_mm_fp16"] = tf
        print("    flag_gems.mm  %7.3f ms  ->  %7.4f TFLOPS" % (t * 1e3, tf))
        if "cublas_fp16" in res:
            print("    相对 cuBLAS: %.3fx" % (tf / res["cublas_fp16"]))
    else:
        print("    flag_gems.mm 不存在, 只有 aten 覆盖")
        t = bench(lambda: torch.mm(a16, b16))
        print("    上下文内 torch.mm %7.3f ms -> %7.4f TFLOPS"
              % (t * 1e3, 2.0 * N**3 / t / 1e12))
except Exception:
    print("    --- FAIL ---")
    traceback.print_exc()

# ---------------------------------------------------------------- fp32 mm
sec("[5] FP32 mm (已知 Triton@CC7.5 有 map::at bug)")
try:
    t = bench(lambda: flag_gems.mm(a32, b32))
    print("    flag_gems.mm fp32  %7.3f ms -> %7.4f TFLOPS"
          % (t * 1e3, 2.0 * N**3 / t / 1e12))
except Exception as e:
    print("    FAIL:", type(e).__name__, str(e)[:300])

# ---------------------------------------------------------------- 算子覆盖
sec("[6] 算子覆盖统计")
try:
    names = [x for x in dir(flag_gems) if not x.startswith("_")]
    print("    flag_gems 顶层可调用符号数:", len(names))
    # 从 aten 注册角度数
    try:
        import flag_gems.ops as _ops
        print("    flag_gems.ops 模块数:", len([x for x in dir(_ops) if not x.startswith("_")]))
    except Exception as e:
        print("    ops 统计失败:", str(e)[:80])

    probe = ["mm", "bmm", "addmm", "softmax", "layer_norm", "rms_norm",
             "gelu", "silu", "relu", "add", "mul", "sub", "div",
             "argmax", "topk", "sort", "mean", "sum", "var_mean",
             "scaled_dot_product_attention", "flash_attention_forward",
             "attention", "embedding", "cross_entropy_loss", "conv2d"]
    have = [x for x in probe if hasattr(flag_gems, x)]
    miss = [x for x in probe if not hasattr(flag_gems, x)]
    print("    有 (%d): %s" % (len(have), ", ".join(have)))
    print("    无 (%d): %s" % (len(miss), ", ".join(miss)))
except Exception:
    traceback.print_exc()

# ---------------------------------------------------------------- 逐算子实测
sec("[7] 逐算子功能 + 性能 (小尺寸, 验证真能跑)")
def bench_op(label, torch_fn, gems_fn, tol=2e-2):
    try:
        r = torch_fn()
        g = gems_fn()
        torch.cuda.synchronize()
        e = (g.float() - r.float()).abs().max().item()
        t_t = bench(torch_fn, iters=20)
        t_g = bench(gems_fn, iters=20)
        ok = "OK " if e <= tol else "ERR"
        print("    %-14s %s  err=%.4g   torch=%7.3fms  gems=%7.3fms  %5.2fx"
              % (label, ok, e, t_t * 1e3, t_g * 1e3, t_t / t_g))
        return True
    except Exception as ex:
        print("    %-14s FAIL  %s: %s" % (label, type(ex).__name__, str(ex)[:110]))
        return False

X = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
W = torch.randn(1024, device="cuda", dtype=torch.float16)

cases = []
if hasattr(flag_gems, "softmax"):
    cases.append(("softmax", lambda: torch.softmax(X, -1), lambda: flag_gems.softmax(X, -1)))
if hasattr(flag_gems, "layer_norm"):
    Nw = torch.randn(1024, device="cuda", dtype=torch.float16)
    Bw = torch.randn(1024, device="cuda", dtype=torch.float16)
    cases.append(("layer_norm", lambda: torch.layer_norm(X, (1024,), Nw, Bw),
                  lambda: flag_gems.layer_norm(X, (1024,), Nw, Bw)))
if hasattr(flag_gems, "rms_norm"):
    cases.append(("rms_norm", lambda: torch.nn.functional.rms_norm(X, (1024,), W),
                  lambda: flag_gems.rms_norm(X, (1024,), W)))
if hasattr(flag_gems, "gelu"):
    cases.append(("gelu", lambda: torch.nn.functional.gelu(X), lambda: flag_gems.gelu(X)))
if hasattr(flag_gems, "silu"):
    cases.append(("silu", lambda: torch.nn.functional.silu(X), lambda: flag_gems.silu(X)))
if hasattr(flag_gems, "add"):
    cases.append(("add", lambda: X + X, lambda: flag_gems.add(X, X)))
if hasattr(flag_gems, "mul"):
    cases.append(("mul", lambda: X * X, lambda: flag_gems.mul(X, X)))
if hasattr(flag_gems, "sum"):
    cases.append(("sum", lambda: torch.sum(X, -1), lambda: flag_gems.sum(X, -1)))

ok_n = 0
for c in cases:
    if bench_op(*c):
        ok_n += 1
print()
print("    小算子通过: %d / %d" % (ok_n, len(cases)))

# ---------------------------------------------------------------- 全局 enable
sec("[8] flag_gems.enable() 全局接管后 torch.mm")
try:
    flag_gems.enable()
    out = torch.mm(a16, b16)
    torch.cuda.synchronize()
    t = bench(lambda: torch.mm(a16, b16))
    tf = 2.0 * N**3 / t / 1e12
    print("    torch.mm 被接管后  %7.3f ms -> %7.4f TFLOPS" % (t * 1e3, tf))
    if "cublas_fp16" in res:
        print("    相对 cuBLAS: %.3fx" % (tf / res["cublas_fp16"]))
    try:
        flag_gems.disable()
        print("    disable() OK")
    except Exception as e:
        print("    disable() 不可用:", str(e)[:80])
except Exception:
    print("    --- FAIL ---")
    traceback.print_exc()

sec("SUMMARY")
for k, v in sorted(res.items()):
    print("    %-22s %8.4f TFLOPS" % (k, v))
print()
print("=" * 72)
