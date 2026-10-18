#!/bin/bash
# P0: 构建完全隔离的 QEMU 测试环境
# 同版本内核 + 最小 initramfs + 我们的模块，无磁盘，崩溃只影响虚拟机
set -u
WORK=/vol1/ganlin/kmod/qemu_test
KERNEL=/boot/vmlinuz-6.18.18.c1126-trim
INITRD=$WORK/test_initrd.cpio.gz

rm -rf $WORK/rootfs
mkdir -p $WORK/rootfs/{bin,dev,proc,sys,tmp}
cd $WORK/rootfs || exit 1

# busybox 静态版
if [ ! -x /bin/busybox.static ]; then
  apt-get install -y busybox-static >/dev/null 2>&1
fi
BB=/bin/busybox
[ -x /bin/busybox.static ] && BB=/bin/busybox.static
echo "使用 busybox: $BB"
cp "$BB" bin/busybox
chmod +x bin/busybox

# 建立 applet 链接
cd bin
for a in sh mount umount insmod rmmod lsmod dmesg cat echo ls poweroff reboot sleep mdev; do
  ln -sf busybox $a
done
cd ..

# 设备节点
mknod dev/console c 5 1 2>/dev/null
mknod dev/null c 1 3 2>/dev/null

# 被测模块
cp /vol1/ganlin/kmod/mx450_probe.ko . 2>/dev/null || echo "警告: 模块不存在"

# init 脚本
cat > init <<'EOF'
#!/bin/sh
/bin/busybox --install -s /bin 2>/dev/null
mount -t proc none /proc
mount -t sysfs none /sys
echo ""
echo "=============================================="
echo "  QEMU 隔离测试环境就绪"
echo "  内核: $(uname -r)"
echo "  内存: $(free -m | awk '/Mem:/{print $2" MB"}')"
echo "=============================================="
echo ""
echo "[1] 模块信息"
ls -l /mx450_probe.ko 2>/dev/null || echo "  模块文件不存在"
echo ""
echo "[2] 尝试加载"
insmod /mx450_probe.ko
RC=$?
echo "  insmod 返回: $RC"
echo ""
echo "[3] 内核日志"
dmesg | tail -12
echo ""
echo "[4] 模块列表"
lsmod | head -5
echo ""
echo "=============================================="
echo "  测试结束，关闭虚拟机"
echo "=============================================="
sync
poweroff -f
EOF
chmod +x init

# 打包
find . -print0 | cpio --null -o -H newc 2>/dev/null | gzip -9 > "$INITRD"
echo "initramfs: $(ls -lh $INITRD | awk '{print $5}')"

# 启动 QEMU（无磁盘、无网络、512MB、串口输dump）
echo ""
echo "=== 启动 QEMU（隔离环境）==="
timeout 90 qemu-system-x86_64 \
  -kernel "$KERNEL" \
  -initrd "$INITRD" \
  -append "console=ttyS0 panic=1" \
  -nographic -no-reboot \
  -m 512 -smp 2 \
  -enable-kvm 2>&1 | tail -40

echo ""
echo "=== QEMU 退出码: $? ==="
