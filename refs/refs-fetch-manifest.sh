#!/bin/bash
# refs-fetch-manifest.sh —— 复现所有外部项目 clone（refs/ 与 B15 相关仓库）
# 用法：bash refs/refs-fetch-manifest.sh
# 各 clone 的 SHA 为本项目实测/钉死版本；GitHub 网络差时本机 clone + 拷贝亦可。
set -u
cd "$(dirname "$0")" || exit 1

fetch() { # name url sha
  local name="$1" url="$2" sha="$3"
  if [ -d "$name/.git" ]; then
    echo "[skip] $name already cloned at $(git -C "$name" rev-parse HEAD)"
  else
    echo "[clone] $name @ $sha"
    git clone --depth 1 "$url" "$name" && \
    git -C "$name" fetch --depth 1 origin "$sha" 2>/dev/null; \
    git -C "$name" checkout -q "$sha" || git -C "$name" checkout -q FETCH_HEAD || true
  fi
}

# ---- 本项目 B 节项目用的外部仓库（钉死 SHA = 本项目实测版）----
fetch Sudoku            https://github.com/scale-snu/Sudoku.git            2c9cc55c72b9ee284181d4eaa3b281a6cffaef29
fetch cmp170hx          https://github.com/Consensus-Protocol/cmp170hx.git d923a25fb2ecb6ee3d62654983f453194b66485f
fetch drama-pp          https://github.com/heechul/drama-pp.git            b5d1f99e6c05ae87a1da8a35352aac768af4bfaf
fetch metaroce          https://github.com/facebookincubator/metaroce.git  8e411134b268066c1a746f1a7ec26d6a10ee1666
fetch open-rdma-driver  https://github.com/open-rdma/open-rdma-driver.git  56fc0565054c995f14d9816389c389a1f8e58a4a
fetch pplx-garden       https://github.com/perplexityai/pplx-garden.git    1ed972ed3f0bd5616c997c9507c25616c63394fc
fetch spdlog            https://github.com/gabime/spdlog.git               7c02e204c92545f869e2f04edaab1f19fe8b19fd

# ---- B15 DeepInfer 相关（DeepSeek 官方，2026-10-20 克隆版）----
mkdir -p b15-deepinfer && cd b15-deepinfer
fetch DeepInfer         https://github.com/deepseek-ai/DeepInfer.git        774ce244b87cc0eb8d4b94ab9f5de2ab22049f01
fetch deepseek-recipe   https://github.com/deepseek-ai/deepseek-recipe.git  935817c66f5d6e64838f419692545f06787e485f
fetch FlashMLA          https://github.com/deepseek-ai/FlashMLA.git        0f11e178da1f67becb087a324b400854931ec3e3
fetch DeepGEMM          https://github.com/deepseek-ai/DeepGEMM.git        5081f8513af57e61696de4f4f04b0e707ffb503e
fetch DeepEP            https://github.com/deepseek-ai/DeepEP.git          b5a33eefda39485f438dc764695b71ba7b9249b6
fetch DeepSelect        https://github.com/deepseek-ai/DeepSelect.git      bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18
cd ..

# ---- NVIDIA 固件二进制获取方法（原件不入仓库，版权属 NVIDIA）----
cat <<'NOTE'
NVIDIA 固件获取（TU117）：
  1) 驱动 570.144 对应包内：/lib/firmware/nvidia/tu117/gsp/*
  2) nouveau 固件提取脚本：refs/cmp170hx 仓库内含 extract-firmware 系列
  3) open-gpu-kernel-modules releases 中的 gsp 固件包
NOTE
echo "ALL_DONE"
