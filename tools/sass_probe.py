#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SASS 机器码级探测
对照组：
  A) kernel_mma  —— 循环体只有 1 条 mma.m16n8k8（Tensor Core 指令，等效 2048 FLOP）
  B) kernel_ffma —— 循环体 32 条 fma.rn.f32（等效 2048 FLOP，CUDA 核路径）
用 NVIDIA 官方 ptxas 编译成 sm_75 cubin，比较 .text 机器码体积。
若 TC 硬件存在 → A 的循环体应远短于 B（1 条 vs 32 条指令）。
若 A 与 B 体积接近 → mma 被编译器展开成 FMA 序列，即 TC 不存在。
"""
import os, struct, subprocess, sys

PTXAS = "/vol1/ganlin/venv-cuda/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin/ptxas"
WORK = "/tmp/sassprobe"
os.makedirs(WORK, exist_ok=True)

# ---------------- A: Tensor Core 路径 ----------------
PTX_A = r"""
.version 7.0
.target sm_75
.address_size 64
.visible .entry kernel_mma(.param .u64 p_ptr, .param .u32 p_iters)
{
	.reg .b32 %r<12>; .reg .f32 %f<8>; .reg .u64 %rd<6>; .reg .pred %p<2>;
	ld.param.u64 %rd1, [p_ptr];
	ld.param.u32 %r1, [p_iters];
	mov.u32 %r2, %tid.x;
	mul.wide.u32 %rd2, %r2, 4;
	add.u64 %rd3, %rd1, %rd2;
	mov.b32 %r3, 0x3C003C00;
	mov.b32 %r4, 0x3C003C00;
	mov.b32 %r5, 0x3C003C00;
	mov.f32 %f1, 0f00000000;
	mov.f32 %f2, 0f00000000;
	mov.f32 %f3, 0f00000000;
	mov.f32 %f4, 0f00000000;
	mov.u32 %r7, %r1;
TL:
	mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f1,%f2,%f3,%f4}, {%r3,%r4}, {%r5}, {%f1,%f2,%f3,%f4};
	sub.u32 %r7, %r7, 1;
	setp.ne.u32 %p1, %r7, 0;
	@%p1 bra TL;
	add.f32 %f1, %f1, %f2;
	add.f32 %f3, %f3, %f4;
	add.f32 %f1, %f1, %f3;
	st.global.f32 [%rd3], %f1;
	ret;
}
"""

# ---------------- B: CUDA 核 FFMA 路径（等量 2048 FLOP） ----------------
PTX_B = r"""
.version 7.0
.target sm_75
.address_size 64
.visible .entry kernel_ffma(.param .u64 p_ptr, .param .u32 p_iters)
{
	.reg .f32 %f<40>; .reg .u32 %r<12>; .reg .u64 %rd<6>; .reg .pred %p<2>;
	ld.param.u64 %rd1, [p_ptr];
	ld.param.u32 %r1, [p_iters];
	mov.u32 %r2, %tid.x;
	mul.wide.u32 %rd2, %r2, 4;
	add.u64 %rd3, %rd1, %rd2;
	mov.f32 %f0, 0f3F800000;
	mov.f32 %f1, 0f00000000;
	mov.f32 %f2, 0f00000000;
	mov.f32 %f3, 0f00000000;
	mov.f32 %f4, 0f00000000;
	mov.u32 %r7, %r1;
TL:
	fma.rn.f32 %f1, %f1, %f0, 0f00000000;
	fma.rn.f32 %f2, %f2, %f0, 0f00000000;
	fma.rn.f32 %f3, %f3, %f0, 0f00000000;
	fma.rn.f32 %f4, %f4, %f0, 0f00000000;
	fma.rn.f32 %f1, %f1, %f0, 0f00000000;
	fma.rn.f32 %f2, %f2, %f0, 0f00000000;
	fma.rn.f32 %f3, %f3, %f0, 0f00000000;
	fma.rn.f32 %f4, %f4, %f0, 0f00000000;
	fma.rn.f32 %f1, %f1, %f0, 0f00000000;
	fma.rn.f32 %f2, %f2, %f0, 0f00000000;
	fma.rn.f32 %f3, %f3, %f0, 0f00000000;
	fma.rn.f32 %f4, %f4, %f0, 0f00000000;
	fma.rn.f32 %f1, %f1, %f0, 0f00000000;
	fma.rn.f32 %f2, %f2, %f0, 0f00000000;
	fma.rn.f32 %f3, %f3, %f0, 0f00000000;
	fma.rn.f32 %f4, %f4, %f0, 0f00000000;
	fma.rn.f32 %f1, %f1, %f0, 0f00000000;
	fma.rn.f32 %f2, %f2, %f0, 0f00000000;
	fma.rn.f32 %f3, %f3, %f0, 0f00000000;
	fma.rn.f32 %f4, %f4, %f0, 0f00000000;
	fma.rn.f32 %f1, %f1, %f0, 0f00000000;
	fma.rn.f32 %f2, %f2, %f0, 0f00000000;
	fma.rn.f32 %f3, %f3, %f0, 0f00000000;
	fma.rn.f32 %f4, %f4, %f0, 0f00000000;
	fma.rn.f32 %f1, %f1, %f0, 0f00000000;
	fma.rn.f32 %f2, %f2, %f0, 0f00000000;
	fma.rn.f32 %f3, %f3, %f0, 0f00000000;
	fma.rn.f32 %f4, %f4, %f0, 0f00000000;
	fma.rn.f32 %f1, %f1, %f0, 0f00000000;
	fma.rn.f32 %f2, %f2, %f0, 0f00000000;
	fma.rn.f32 %f3, %f3, %f0, 0f00000000;
	fma.rn.f32 %f4, %f4, %f0, 0f00000000;
	sub.u32 %r7, %r7, 1;
	setp.ne.u32 %p1, %r7, 0;
	@%p1 bra TL;
	add.f32 %f1, %f1, %f2;
	add.f32 %f3, %f3, %f4;
	add.f32 %f1, %f1, %f3;
	st.global.f32 [%rd3], %f1;
	ret;
}
"""

def parse_elf_sections(path):
    data = open(path, "rb").read()
    if data[:4] != b"\x7fELF":
        return {}
    e_shoff    = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize= struct.unpack_from("<H", data, 0x3A)[0]
    e_shnum    = struct.unpack_from("<H", data, 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", data, 0x3E)[0]
    sh = e_shoff + e_shstrndx * e_shentsize
    str_off = struct.unpack_from("<Q", data, sh+0x18)[0]
    strtab = data[str_off:str_off+struct.unpack_from("<Q", data, sh+0x20)[0]]
    out = {}
    for i in range(e_shnum):
        o = e_shoff + i*e_shentsize
        n_off = struct.unpack_from("<I", data, o)[0]
        size  = struct.unpack_from("<Q", data, o+0x20)[0]
        nm = strtab[n_off:].split(b"\x00")[0].decode(errors="replace")
        out[nm] = size
    return out

def build(name, ptx, arch="sm_75"):
    p = os.path.join(WORK, name + ".ptx")
    c = os.path.join(WORK, name + ".cubin")
    open(p, "w").write(ptx)
    r = subprocess.run([PTXAS, "-arch=" + arch, "-v", p, "-o", c],
                       capture_output=True, text=True)
    print("--- %s (%s) ---" % (name, arch))
    if r.returncode != 0:
        print("  编译失败:", r.stderr.strip()[:300])
        return None, None
    for line in r.stderr.splitlines():
        if "Used" in line or "Compiling" in line or "error" in line:
            print("  " + line.strip())
    secs = parse_elf_sections(c)
    text = secs.get(".text", 0)
    tot  = os.path.getsize(c)
    print("  cubin 总大小 : %d B" % tot)
    print("  .text 代码段 : %d B  → 约 %d 条 SASS 指令（16B/条）" % (text, text // 16))
    return text, tot

print("=" * 66)
print("SASS 机器码对比实验  (ptxas 12.9, target sm_75)")
print("=" * 66)
ta, sa = build("kernel_mma", PTX_A)
tb, sb = build("kernel_ffma", PTX_B)

print("\n" + "=" * 66)
if ta and tb:
    print("mma 内核 .text  : %6d B" % ta)
    print("ffma 内核 .text : %6d B" % tb)
    print("比值 (ffma/mma) : %.2f×" % (tb / ta))
    print("-" * 66)
    if tb / ta > 5:
        print("结论：mma 编译后显著更短 → 生成的是真 Tensor Core 指令（HMMA）")
    else:
        print("结论：两者体积接近 → mma 被展开成 FMA 序列，该架构无可用 TC 路径")
print("=" * 66)
