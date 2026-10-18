#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensor Core 吞吐终审：用 8 路独立累加器 + 大量 warp 并行，
彻底排除"延迟受限"和"依赖链"两个干扰变量。
若 TC 硬件可用，8 路 × 8 warp 应能打满 ~6-12 TFLOPS。
"""
import ctypes, time

cu = ctypes.CDLL("libcuda.so.1")
def chk(r, n=""):
    if r != 0: raise RuntimeError("CUDA err %d @ %s" % (r, n))
cu.cuInit.argtypes=[ctypes.c_uint]; chk(cu.cuInit(0))
d=ctypes.c_int(); chk(cu.cuDeviceGet(ctypes.byref(d),0))
c=ctypes.c_void_p(); chk(cu.cuCtxCreate_v2(ctypes.byref(c),0,d))

PTX = r"""
.version 7.0
.target sm_75
.address_size 64
.visible .entry tc8(.param .u64 p_ptr, .param .u32 p_iters)
{
	.reg .b32 %r<16>; .reg .f32 %f<40>; .reg .u64 %rd<6>; .reg .pred %p<2>;
	ld.param.u64 %rd1, [p_ptr];
	ld.param.u32 %r1, [p_iters];
	mov.u32 %r2, %tid.x;
	mul.wide.u32 %rd2, %r2, 4;
	add.u64 %rd3, %rd1, %rd2;
	mov.b32 %r3, 0x3C003C00;
	mov.b32 %r4, 0x3C003C00;
	mov.b32 %r5, 0x3C003C00;

	mov.f32 %f1, 0f00000000;  mov.f32 %f2, 0f00000000;
	mov.f32 %f3, 0f00000000;  mov.f32 %f4, 0f00000000;
	mov.f32 %f5, 0f00000000;  mov.f32 %f6, 0f00000000;
	mov.f32 %f7, 0f00000000;  mov.f32 %f8, 0f00000000;
	mov.f32 %f9, 0f00000000;  mov.f32 %f10, 0f00000000;
	mov.f32 %f11, 0f00000000; mov.f32 %f12, 0f00000000;
	mov.f32 %f13, 0f00000000; mov.f32 %f14, 0f00000000;
	mov.f32 %f15, 0f00000000; mov.f32 %f16, 0f00000000;
	mov.f32 %f17, 0f00000000; mov.f32 %f18, 0f00000000;
	mov.f32 %f19, 0f00000000; mov.f32 %f20, 0f00000000;
	mov.f32 %f21, 0f00000000; mov.f32 %f22, 0f00000000;
	mov.f32 %f23, 0f00000000; mov.f32 %f24, 0f00000000;
	mov.f32 %f25, 0f00000000; mov.f32 %f26, 0f00000000;
	mov.f32 %f27, 0f00000000; mov.f32 %f28, 0f00000000;
	mov.f32 %f29, 0f00000000; mov.f32 %f30, 0f00000000;
	mov.f32 %f31, 0f00000000; mov.f32 %f32, 0f00000000;

	mov.u32 %r7, %r1;
TL:
	mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f1,%f2,%f3,%f4},     {%r3,%r4}, {%r5}, {%f1,%f2,%f3,%f4};
	mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f5,%f6,%f7,%f8},     {%r3,%r4}, {%r5}, {%f5,%f6,%f7,%f8};
	mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f9,%f10,%f11,%f12},  {%r3,%r4}, {%r5}, {%f9,%f10,%f11,%f12};
	mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f13,%f14,%f15,%f16}, {%r3,%r4}, {%r5}, {%f13,%f14,%f15,%f16};
	mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f17,%f18,%f19,%f20}, {%r3,%r4}, {%r5}, {%f17,%f18,%f19,%f20};
	mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f21,%f22,%f23,%f24}, {%r3,%r4}, {%r5}, {%f21,%f22,%f23,%f24};
	mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f25,%f26,%f27,%f28}, {%r3,%r4}, {%r5}, {%f25,%f26,%f27,%f28};
	mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f29,%f30,%f31,%f32}, {%r3,%r4}, {%r5}, {%f29,%f30,%f31,%f32};
	sub.u32 %r7, %r7, 1;
	setp.ne.u32 %p1, %r7, 0;
	@%p1 bra TL;

	add.f32 %f1, %f1, %f2;    add.f32 %f3, %f3, %f4;
	add.f32 %f5, %f5, %f6;    add.f32 %f7, %f7, %f8;
	add.f32 %f9, %f9, %f10;   add.f32 %f11, %f11, %f12;
	add.f32 %f13, %f13, %f14; add.f32 %f15, %f15, %f16;
	add.f32 %f17, %f17, %f18; add.f32 %f19, %f19, %f20;
	add.f32 %f21, %f21, %f22; add.f32 %f23, %f23, %f24;
	add.f32 %f25, %f25, %f26; add.f32 %f27, %f27, %f28;
	add.f32 %f29, %f29, %f30; add.f32 %f31, %f31, %f32;
	add.f32 %f1, %f1, %f3;    add.f32 %f5, %f5, %f7;
	add.f32 %f9, %f9, %f11;   add.f32 %f13, %f13, %f15;
	add.f32 %f17, %f17, %f19; add.f32 %f21, %f21, %f23;
	add.f32 %f25, %f25, %f27; add.f32 %f29, %f29, %f31;
	add.f32 %f1, %f1, %f5;    add.f32 %f9, %f9, %f13;
	add.f32 %f17, %f17, %f21; add.f32 %f25, %f25, %f29;
	add.f32 %f1, %f1, %f9;    add.f32 %f17, %f17, %f25;
	add.f32 %f1, %f1, %f17;
	st.global.f32 [%rd3], %f1;
	ret;
}
"""
m=ctypes.c_void_p(); chk(cu.cuModuleLoadData(ctypes.byref(m), PTX.encode()+b"\x00"))
fn=ctypes.c_void_p(); chk(cu.cuModuleGetFunction(ctypes.byref(fn), m, b"tc8"))
cu.cuMemAlloc_v2.argtypes=[ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
cu.cuMemsetD8_v2.argtypes=[ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t]

B,G = 256, 2048
p=ctypes.c_uint64(); chk(cu.cuMemAlloc_v2(ctypes.byref(p), B*G*4)); cu.cuMemsetD8_v2(p,0,B*G*4)

def run(it):
    a0=ctypes.c_uint64(p.value); a1=ctypes.c_uint32(it)
    arr=(ctypes.c_void_p*2)()
    arr[0]=ctypes.cast(ctypes.byref(a0),ctypes.c_void_p)
    arr[1]=ctypes.cast(ctypes.byref(a1),ctypes.c_void_p)
    chk(cu.cuLaunchKernel(fn,G,1,1,B,1,1,0,None,arr,None))
    chk(cu.cuCtxSynchronize())

run(200)
print("="*62)
print("Tensor Core 吞吐（8 路独立累加器 / 8 warp per block / 2048 blocks）")
print("="*62)
best=0
for it in (10000, 40000, 40000):
    t0=time.time(); run(it); dt=time.time()-t0
    flops = 2048.0*(B/32)*8*G*it
    tf = flops/dt/1e12
    best=max(best,tf)
    print("  it=%-6d  %6.3fs  ->  %5.2f TFLOPS" % (it, dt, tf))
print("-"*62)
print("  最高 %.2f TFLOPS" % best)
print("  Turing TU117 理论 TC 峰值 ≈ 6–12 TFLOPS")
print("  FP32 CUDA 核实测 3.06 / FP16x2 实测 5.77 TFLOPS")
print("="*62)
