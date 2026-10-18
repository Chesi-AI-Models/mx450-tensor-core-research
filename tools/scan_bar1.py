#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 BAR1 全部 256MB，找出有数据的区间"""
import os, mmap

DEV = "/sys/bus/pci/devices/0000:2c:00.0/resource1"
SEG = 4 * 1024 * 1024      # 4 MB
TOTAL = 256 * 1024 * 1024  # 256 MB

fd = os.open(DEV, os.O_RDONLY | os.O_SYNC)
print("open OK, 总大小 256 MB，按 4 MB 分段扫描")
print("=" * 60)
print("%-16s %-10s %s" % ("偏移", "非零字节", "占比"))
print("-" * 60)

hot = []
for off in range(0, TOTAL, SEG):
    try:
        mm = mmap.mmap(fd, SEG, mmap.MAP_SHARED, mmap.PROT_READ, offset=off)
        d = mm[:SEG]
        mm.close()
        nz = sum(1 for b in d if b)
        if nz:
            hot.append((off, nz, d))
            print("0x%08X       %-10d %.2f%%" % (off, nz, nz*100.0/SEG))
    except Exception as e:
        print("0x%08X       读取失败: %s" % (off, e))
        break

os.close(fd)
print("-" * 60)
print("有数据的段数: %d / %d" % (len(hot), TOTAL // SEG))

if hot:
    # 保存第一段有数据的区域
    off, nz, d = hot[0]
    p = "/vol1/ganlin/vram/bar1_hot.bin"
    open(p, "wb").write(d)
    print("首个有数据段已保存: %s (偏移 0x%X, 非零 %.1f%%)" % (p, off, nz*100.0/SEG))
else:
    print("BAR1 全 256MB 均为零 —— 该映射窗口未暴露实际显存内容")
