#!/bin/bash
# B7: 在 NAS 上真跑 DRAMA++ measure
set -u
D=/vol1/ganlin/refs/drama-pp/re
OUT=/vol1/ganlin/refs/drama-pp/run
mkdir -p $OUT
cd $D || exit 1

echo "=== [0] 跑前系统状态 ==="
free -g | awk '/^Mem:/{printf "  mem %sG total / %sG avail\n", $2, $7}'
lsmod | grep -c greenboost | xargs -I{} echo "  greenboost 模块数: {}"

echo
echo "=== [1] 探测系统内存配置 ==="
echo "--- dmidecode 内存条 ---"
dmidecode -t memory 2>/dev/null | grep -E "Size:|Type:|Speed:|Rank|Locator:" | grep -v "No Module" | head -20
echo "--- CPU ---"
grep -m1 "model name" /proc/cpuinfo
echo "  core ids: $(cat /sys/devices/system/cpu/cpu*/topology/core_id 2>/dev/null | sort -u | tr '\n' ' ')"
echo "  threads: $(nproc)"

echo
echo "=== [2] 运行 measure -s 16 -g 1 (16 sets, 1GB) ==="
cd $OUT || exit 1
cp $D/measure . 2>/dev/null
chmod +x measure
/usr/bin/time -v ./measure -s 16 -g 1 -c 0 2>&1 | tail -40
echo "  measure rc=$?"

echo
echo "=== [3] 产物 ==="
ls -la $OUT/
echo
echo "--- map.txt ---"
cat $OUT/map.txt 2>/dev/null || echo "  (无 map.txt)"
echo
echo "--- set*.txt 数量 ---"
ls $OUT/set*.txt 2>/dev/null | wc -l

echo
echo "=== [4] 跑后状态 ==="
free -g | awk '/^Mem:/{printf "  mem %sG total / %sG avail\n", $2, $7}'
