#!/bin/bash
# 通过 PRAMIN 窗口读取 GPU 显存（BAR0 + 0x700000，1MB 滑动窗口）
# 全程只读，结束后必定恢复 nvidia 驱动
set -u
DEV=/sys/bus/pci/devices/0000:2c:00.0

echo "=== 卸载 nvidia 驱动 ==="
for m in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
  modprobe -r "$m" 2>/dev/null && echo "  unloaded $m" || echo "  skip $m"
done
sleep 1

echo "=== 通过 PRAMIN 读显存 ==="
python3 - <<'PYEOF'
import os, struct, mmap

DEV = "/sys/bus/pci/devices/0000:2c:00.0"
BAR0_WINDOW = 0x1700      # NV_PBUS_BAR0_WINDOW
PRAMIN_BASE = 0x700000    # PRAMIN aperture
PRAMIN_SIZE = 0x100000    # 1MB

# 打开 BAR0（映射 8MB 以覆盖 PRAMIN）
fd = os.open(DEV + "/resource0", os.O_RDONLY | os.O_SYNC)
size = 0x800000
try:
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
    print("BAR0 mmap OK: %d bytes" % size)

    win = struct.unpack("<I", mm[BAR0_WINDOW:BAR0_WINDOW+4])[0]
    print("\n[NV_PBUS_BAR0_WINDOW @0x1700] = 0x%08X" % win)
    print("  TARGET    = %d  (0=VRAM, 1=SYS_COH, 2=SYS_NONCOH)" % ((win >> 24) & 0x3))
    print("  BASE_ADDR = 0x%06X  (VRAM 地址 bits39:16)" % (win & 0xFFFFFF))
    print("  => 窗口当前覆盖 VRAM 0x%010X 起" % ((win & 0xFFFFFF) << 16))

    print("\n[PRAMIN 窗口前 256 字节]")
    start = 0
    nz = 0
    for off in range(0, 256, 16):
        chunk = mm[PRAMIN_BASE+off : PRAMIN_BASE+off+16]
        vals = " ".join("%02x" % b for b in chunk)
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print("  +0x%04X  %s  |%s|" % (off, vals, asc))
        nz += sum(1 for b in chunk if b)

    # 统计整个窗口的非零分布（按 64KB 分段）
    print("\n[1MB 窗口非零分布，按 64KB 分段]")
    for seg in range(0, PRAMIN_SIZE, 0x10000):
        chunk = mm[PRAMIN_BASE+seg : PRAMIN_BASE+seg+0x10000]
        n = sum(1 for b in chunk if b)
        if n:
            print("  +0x%05X  %6d / 65536 非零 (%.1f%%)" % (seg, n, n*100.0/65536))
    print("  前 256 字节非零字节数: %d / 256" % nz)

    mm.close()
except Exception as e:
    print("失败:", e)
finally:
    os.close(fd)
PYEOF

echo ""
echo "=== 恢复 nvidia 驱动 ==="
for m in nvidia nvidia_modeset nvidia_drm nvidia_uvm; do
  modprobe "$m" 2>/dev/null && echo "  loaded $m"
done
sleep 1
nvidia-smi -L 2>&1
