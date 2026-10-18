#!/bin/bash
# 用 PRAMIN 窗口扫描 VRAM：移动 NV_PBUS_BAR0_WINDOW 遍历显存，找有数据的区域
# 只写窗口寄存器（标准操作，NVIDIA 驱动自己也这么做），读完恢复原值
set -u
DEV=/sys/bus/pci/devices/0000:2c:00.0

echo "=== 卸载 nvidia 驱动 ==="
for m in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
  modprobe -r "$m" 2>/dev/null && echo "  unloaded $m" || echo "  skip $m"
done
sleep 1

python3 - <<'PYEOF'
import os, struct, mmap

DEV = "/sys/bus/pci/devices/0000:2c:00.0"
WIN = 0x1700
PRAMIN = 0x700000
STEP = 0x100000          # 1MB 窗口
SCAN_LIMIT = 256 * 0x100000   # 扫前 256MB
SAMPLE = 0x1000          # 每窗口采样 4KB

fd = os.open(DEV + "/resource0", os.O_RDWR | os.O_SYNC)
mm = mmap.mmap(fd, 0x800000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
print("BAR0 mmap RW OK")

orig = struct.unpack("<I", mm[WIN:WIN+4])[0]
print("原 WINDOW = 0x%08X" % orig)

def set_window(addr, target=0):
    val = ((target & 0x3) << 24) | ((addr >> 16) & 0xFFFFFF)
    mm[WIN:WIN+4] = struct.pack("<I", val)
    return val

print("\n扫描前 %d MB (每 1MB 采样 4KB)..." % (SCAN_LIMIT // 0x100000))
print("%-14s %-10s %s" % ("VRAM 偏移", "非零字节", "占比"))
print("-" * 46)

hits = []
for off in range(0, SCAN_LIMIT, STEP):
    set_window(off)
    chunk = mm[PRAMIN:PRAMIN+SAMPLE]
    n = sum(1 for b in chunk if b)
    if n > 16:                    # 超过 16 个非零字节才算有内容
        hits.append((off, n))
        print("0x%09X    %-10d %.1f%%" % (off, n, n*100.0/SAMPLE))

print("-" * 46)
print("有数据的区域: %d / %d" % (len(hits), SCAN_LIMIT // STEP))

if hits:
    off = hits[0][0]
    set_window(off)
    print("\n首个有数据窗口 0x%09X 前 128 字节:" % off)
    chunk = mm[PRAMIN:PRAMIN+128]
    for i in range(0, 128, 16):
        seg = chunk[i:i+16]
        print("  +0x%04X  %s  |%s|" % (
            i, " ".join("%02x" % b for b in seg),
            "".join(chr(b) if 32 <= b < 127 else "." for b in seg)))

# 恢复原窗口
mm[WIN:WIN+4] = struct.pack("<I", orig)
print("\nWINDOW 已恢复为 0x%08X" % orig)
mm.close(); os.close(fd)
PYEOF

echo ""
echo "=== 恢复 nvidia 驱动 ==="
for m in nvidia nvidia_modeset nvidia_drm nvidia_uvm; do
  modprobe "$m" 2>/dev/null && echo "  loaded $m"
done
sleep 1
nvidia-smi -L 2>&1
