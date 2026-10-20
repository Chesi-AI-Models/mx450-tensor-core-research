#!/bin/bash
# B7: 在 NAS 上编译 DRAMA++
set -u
D=/vol1/ganlin/refs/drama-pp/re
cd $D || { echo "目录不存在"; exit 1; }

echo "=== [1] 修正权限 ==="
chmod 644 $D/*.cpp $D/*.h $D/Makefile $D/*.py 2>/dev/null
chown -R root:root $D 2>/dev/null
ls -la

echo
echo "=== [2] Makefile 内容 ==="
cat Makefile

echo
echo "=== [3] 编译 ==="
make 2>&1 | tail -30
RC=$?
echo "  make rc=$RC"

echo
echo "=== [4] 产物 ==="
if [ -f measure ]; then
    ls -la measure
    file measure 2>/dev/null
else
    echo "  !! measure 未生成"
fi

echo
echo "=== [5] 环境 ==="
echo "  nproc = $(nproc)"
free -g | awk '/^Mem:/{printf "  mem = %sG total, %sG avail\n", $2, $7}'
echo "  pagemap 可读? $([ -r /proc/self/pagemap ] && echo yes || echo no)"
echo "  THP = $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
echo "  g++ = $(g++ --version 2>/dev/null | head -1)"
