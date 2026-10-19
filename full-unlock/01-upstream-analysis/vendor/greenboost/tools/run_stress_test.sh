#!/bin/bash
# GreenBoost v2.2 — stress test runner
# Usage: sudo ./run_stress_test.sh [SIZE_GB]
#   SIZE_GB defaults to 63 (fills T1+T2)
#   Use > 63 to trigger T3 NVMe spillover

set -e

echo "=========================================="
echo "GreenBoost v2.2 — 3-Tier Stress Test"
echo "  T1 VRAM : 12 GB  RTX 5070"
echo "  T2 System DDR : 51 GB  pool"
echo "  T3 NVMe : 576 GB swap"
echo "=========================================="
echo ""

if ! lsmod | grep -q "^greenboost "; then
    echo "ERROR: greenboost module not loaded"
    echo "Load it: sudo insmod greenboost.ko"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "stress_test_greenboost" ] || [ "stress_test_greenboost.c" -nt "stress_test_greenboost" ]; then
    echo "Compiling stress_test_greenboost..."
    gcc -o stress_test_greenboost stress_test_greenboost.c -lpthread || {
        echo "ERROR: compilation failed"
        exit 1
    }
    echo "OK: compiled"
fi

SIZE=${1:-63}
echo "Target size : ${SIZE} GB"
if [ "$SIZE" -gt 63 ]; then
    echo "Mode        : T1+T2+T3 (NVMe spillover active)"
else
    echo "Mode        : T1+T2 only"
fi
echo "Press Ctrl+C to stop"
echo ""

sudo ./stress_test_greenboost "$SIZE"

echo ""
echo "Test done"
