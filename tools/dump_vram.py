#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""尝试从 GPU 显存（BAR1）读取数据，寻找 FECS/GPCCS 微码"""
import os, sys, mmap, errno

DEV = "/sys/bus/pci/devices/0000:2c:00.0"
BAR1_PHYS = 0x6020000000
BAR1_SIZE = 0x10000000          # 256 MB
READ_SIZE = 0x100000            # 先读 1 MB

print("=" * 66)

# 方法 1: mmap resource1
print("[方法1] mmap %s/resource1" % DEV)
try:
    fd = os.open(DEV + "/resource1", os.O_RDONLY | os.O_SYNC)
    print("  open OK (fd=%d)" % fd)
    try:
        mm = mmap.mmap(fd, READ_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
        data = mm[:READ_SIZE]
        mm.close()
        nz = sum(1 for b in data if b)
        print("  mmap OK  读到 %d 字节，非零 %d (%.1f%%)" % (len(data), nz, nz*100.0/len(data)))
        open("/vol1/ganlin/vram/bar1_head.bin", "wb").write(data)
        print("  已保存 /vol1/ganlin/vram/bar1_head.bin")
        ok = True
    except Exception as e:
        print("  mmap 失败: %s (errno=%s)" % (e, getattr(e, 'errno', '?')))
        ok = False
    os.close(fd)
except Exception as e:
    print("  open 失败: %s" % e)
    ok = False

# 方法 2: /dev/mem + lseek + read
print("\n[方法2] /dev/mem + lseek(0x%X) + read" % BAR1_PHYS)
try:
    fd = os.open("/dev/mem", os.O_RDONLY | os.O_SYNC)
    print("  open OK (fd=%d)" % fd)
    try:
        os.lseek(fd, BAR1_PHYS, os.SEEK_SET)
        data = os.read(fd, READ_SIZE)
        nz = sum(1 for b in data if b)
        print("  read OK  %d 字节，非零 %d (%.1f%%)" % (len(data), nz, nz*100.0/max(len(data),1)))
        if data:
            open("/vol1/ganlin/vram/bar1_devmem.bin", "wb").write(data)
            print("  已保存 /vol1/ganlin/vram/bar1_devmem.bin")
    except Exception as e:
        print("  lseek/read 失败: %s (errno=%s)" % (e, getattr(e, 'errno', '?')))
    os.close(fd)
except Exception as e:
    print("  open 失败: %s" % e)

print("=" * 66)
