#!/bin/bash
# 直接 mmap GPU BAR0 读取 PMC 寄存器区（只读，安全）
# 结束后必定恢复 nvidia 驱动
set -u
DEV=/sys/bus/pci/devices/0000:2c:00.0

echo "=== 卸载 nvidia 驱动 ==="
for m in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
  modprobe -r "$m" 2>/dev/null && echo "unloaded $m" || echo "skip $m"
done
sleep 1

echo "=== BAR 资源 ==="
cat $DEV/resource | head -8

echo "=== 读取 PMC 寄存器区 ==="
python3 - <<'PYEOF'
import os, struct, mmap

DEV = "/sys/bus/pci/devices/0000:2c:00.0"
line = open(DEV + "/resource").read().splitlines()[0].split()
start, end, flags = int(line[0],16), int(line[1],16), int(line[2],16)
size = end - start + 1
print("BAR0: 0x%08x - 0x%08x  size = %.1f MB  flags=0x%x" % (start, end, size/1048576, flags))

try:
    fd = os.open(DEV + "/resource0", os.O_RDONLY | os.O_SYNC)
except Exception as e:
    print("open resource0 failed:", e); raise SystemExit(1)

try:
    mm = mmap.mmap(fd, 0x2000, mmap.MAP_SHARED, mmap.PROT_READ)
except Exception as e:
    print("mmap failed:", e); os.close(fd); raise SystemExit(1)

print("\n--- 非零寄存器（PMC 前 8KB）---")
found = 0
for off in range(0, 0x2000, 4):
    v = struct.unpack("<I", mm[off:off+4])[0]
    if v not in (0, 0xFFFFFFFF):
        print("  0x%05x = 0x%08x" % (off, v))
        found += 1
print("  共 %d 个非零寄存器" % found)

b0 = struct.unpack("<I", mm[0x00:0x04])[0]
print("\n--- NV_PMC_BOOT_0 解析 ---")
print("  原始值 = 0x%08x" % b0)
print("  chipset  = 0x%03x" % ((b0 >> 20) & 0xFFF))
print("  revision = 0x%02x" % ((b0 >> 16) & 0xFF))
print("  family   = 0x%02x" % ((b0 >> 8) & 0xFF))
print("  minor    = 0x%02x" % (b0 & 0xFF))

b1 = struct.unpack("<I", mm[0x04:0x08])[0]
print("\n  BOOT_1 = 0x%08x" % b1)

mm.close(); os.close(fd)
PYEOF

RC=$?

echo "=== 恢复 nvidia 驱动 ==="
for m in nvidia nvidia_modeset nvidia_drm nvidia_uvm; do
  modprobe "$m" 2>/dev/null && echo "loaded $m" || echo "skip $m"
done

echo "=== 最终状态 ==="
nvidia-smi -L 2>&1
exit $RC
