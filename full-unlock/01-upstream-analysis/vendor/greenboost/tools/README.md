# GreenBoost — Model Optimization Tools

These tools extend GreenBoost beyond memory overflow into active model optimization.
The goal is to reduce model size until it fits comfortably in T1+T2, using GreenBoost's
System DDR pool for KV cache only — which gives the best possible performance on consumer hardware.

---

## Bundled libraries

| Tool | Purpose |
|------|---------|
| **ExLlamaV3** | High-performance inference engine with a native GreenBoost KV cache layer. KV tensors are allocated directly from `/dev/greenboost` as zero-copy mmap → numpy → PyTorch tensors. No CUDA shim involved for cache I/O. |
| **kvpress** | Runtime KV cache compression (ExpAttn, SnapKV, KnormPress). Applies at inference time, no retraining needed. Reduces System DDR pressure significantly at long context. |
| **NVIDIA ModelOpt** | Post-training quantization (FP8, INT4-AWQ, NVFP4). No retraining — only a calibration pass (~512 samples). Cuts the 32 GB model to ~16 GB (FP8) or ~8 GB (INT4-AWQ). |
| **TensorRT-Edge-LLM** | TRT engine export for desktop RTX GPUs. |
| **Unsloth + LoRA** | Parameter-efficient fine-tuning that fits a 30B model into 12 GB VRAM via 4-bit base quantization and rank-16 adapters. Requires your own training data. |

---

## Usage

```bash
# ExLlamaV3 with GreenBoost System DDR KV cache offload:
./tools/greenboost-exllama.sh --model /opt/models/glm-4.7-flash-exl3

# Convert a HuggingFace model to EXL3 format (4bpw → ~16 GB):
./tools/convert_to_exl3.sh --model THUDM/glm-4.7-flash-hf --bpw 4.0

# FP8 quantization (32 GB → ~16 GB, no retraining):
python tools/greenboost-ptq.py --model THUDM/glm-4.7-flash-hf --quant fp8

# INT4-AWQ quantization (32 GB → ~8 GB):
python tools/greenboost-ptq.py --model THUDM/glm-4.7-flash-hf --quant int4-awq

# Runtime KV cache compression benchmark:
./tools/greenboost-kvpress.sh --model THUDM/glm-4.7-flash-hf --benchmark

# Run all optimization strategies in one pass (via setup script):
sudo ./greenboost_setup.sh optimize-model --model THUDM/glm-4.7-flash-hf --strategy all
```

---

## Observed performance (glm-4.7-flash:q8_0, RTX 5070)

| Setup | Decode tok/s | TTFT |
|-------|-------------|------|
| Ollama + GreenBoost shim (UVM path) | 3–8 | 4–12 s |
| + kvpress 50% KV compression | 4–8 | 3–10 s |
| ExLlamaV3 + GreenBoost cache | 8–20 | 2–8 s |
| ModelOpt FP8 (16 GB model) | 10–25 | 1–5 s |
| ExLlamaV3 EXL3 2bpw (8 GB, full VRAM) | 25–60 | 0.5–2 s |

The best strategy is to shrink the model until it fits in VRAM — either with EXL3
quantization or ModelOpt PTQ — and use GreenBoost's System DDR pool for KV cache only.
The PCIe link is the hard bottleneck when weights overflow VRAM (~32 GB/s on PCIe 4 x16, ~64 GB/s on PCIe 5 x16).

---

## ExLlamaV3 setup

ExLlamaV3 is installed automatically by `full-install` into `/opt/greenboost/venv`.
To activate manually:

```bash
source /opt/greenboost/venv/bin/activate
python -c "import exllamav3; print('ExLlamaV3 OK')"
```

The GreenBoost KV cache layer (`CacheLayer_greenboost`) allocates KV tensors directly
from `/dev/greenboost` without going through the CUDA shim — lower overhead and more
predictable bandwidth than the UVM overflow path.

---

## LoRA fine-tuning

```bash
# Fine-tune with your own data (fits in 12 GB VRAM via 4-bit base + rank-16 adapters):
python tools/greenboost-lora-train.py \
  --model THUDM/glm-4.7-flash-hf \
  --data your_dataset.jsonl \
  --rank 16 \
  --output /opt/models/glm-4.7-flash-lora
```

---

## Stress test

```bash
# Run memory tier stress test (verifies T1→T2→T3 overflow chain):
./tools/run_stress_test.sh
# or directly:
python gb_pool_stress.py
```
