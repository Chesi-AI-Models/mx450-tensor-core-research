#!/bin/bash
# 建独立 venv, 装 torch(cu126, 自带 triton 3.x) + flag_gems, 与主环境完全隔离
set -x
VENV=/vol1/ganlin/venv_t3
PY=/usr/bin/python3

$PY -m venv $VENV || exit 1
$VENV/bin/pip install -q --upgrade pip setuptools wheel

echo "=== installing torch (cu126) ==="
$VENV/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126 || exit 2

echo "=== installing flag_gems ==="
$VENV/bin/pip install --no-cache-dir --no-deps flag_gems || exit 3
$VENV/bin/pip install --no-cache-dir --no-deps sqlalchemy greenlet || exit 4

echo "=== VERSIONS ==="
$VENV/bin/python -c "import torch,triton;print('torch',torch.__version__);print('triton',triton.__version__);print('cuda_avail',torch.cuda.is_available());print('cap',torch.cuda.get_device_capability(0))"

echo "=== IMPORT flag_gems ==="
$VENV/bin/python -c "import flag_gems;print('flag_gems OK',flag_gems.__version__)"

echo "=== DONE ==="
