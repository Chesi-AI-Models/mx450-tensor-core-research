# GreenBoost — integration guide for inference tools

## How GreenBoost hooks in

GreenBoost works by intercepting `cudaMalloc` and related CUDA symbols via `LD_PRELOAD`,
plus intercepting `dlsym` so that virtual VRAM figures (63 GB) are returned to any app
that queries VRAM size at runtime through `dlopen`+`dlsym`.

For this to work the shim must be loaded before CUDA is touched. All tools below load
CUDA lazily (via PyTorch, ctypes, or their own dlopen), so `GREENBOOST_ACTIVE=1` is
required in addition to `LD_PRELOAD`:

```bash
export LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so
export GREENBOOST_ACTIVE=1
```

Or use the installed wrapper (sets both automatically):

```bash
greenboost run python your_script.py
greenboost run python -m vllm.entrypoints.openai.api_server ...
```

---

## Ollama

Handled automatically by `install-sys-configs`, which injects the shim and
`GREENBOOST_ACTIVE=1` into the systemd unit. No manual wrapper needed.

```bash
ollama run glm-4.7-flash:q8_0   # GreenBoost is transparent
```

If running Ollama outside systemd:

```bash
greenboost run ollama serve
```

---

## vLLM

vLLM loads libcuda lazily through PyTorch (`torch.cuda` → `ctypes.CDLL`).

```bash
greenboost run python -m vllm.entrypoints.openai.api_server \
    --model /opt/models/glm-4.7-flash-hf \
    --dtype float16 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95
```

As a systemd service, add to the unit:

```ini
[Service]
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
Environment="GREENBOOST_ACTIVE=1"
```

---

## PyTorch scripts

Any `import torch; torch.cuda.*` call triggers lazy loading of libcuda via ctypes.

```bash
greenboost run python your_inference_script.py
```

Or inline without the wrapper:

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python your_script.py
```

---

## text-generation-inference (TGI)

TGI uses a PyTorch backend — same lazy-CUDA pattern:

```bash
greenboost run text-generation-launcher \
    --model-id /opt/models/glm-4.7-flash-hf \
    --num-shard 1 \
    --max-total-tokens 131072
```

As a systemd service:

```ini
[Service]
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
Environment="GREENBOOST_ACTIVE=1"
```

---

## CTranslate2

CTranslate2 loads libcudart via ctypes in its Python bindings:

```bash
greenboost run python your_ctranslate2_script.py
```

---

## Hugging Face Transformers

Transformers loads CUDA through PyTorch — same lazy-CUDA pattern as vLLM and TGI.

```bash
greenboost run python your_transformers_script.py
```

With `device_map="auto"`, Transformers queries available VRAM before placing layers.
GreenBoost reports 63 GB, so the full model is placed on the "GPU" (T1+T2) instead of
being split to CPU:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/opt/models/glm-4.7-flash-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",          # sees 63 GB — loads entire model onto T1+T2
)
tokenizer = AutoTokenizer.from_pretrained("/opt/models/glm-4.7-flash-hf")

messages = [{"role": "user", "content": "Hello!"}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
).to(model.device)

output = model.generate(input_ids, max_new_tokens=300)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Pipeline API works the same way:

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="/opt/models/glm-4.7-flash-hf",
    torch_dtype="bfloat16",
    device_map="auto",
)
print(pipe("Hello!", max_new_tokens=200)[0]["generated_text"])
```

Downloading a model then running it in one shot:

```bash
greenboost run python - <<'EOF'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

snapshot_download("THUDM/glm-4.7-flash-hf", local_dir="/opt/models/glm-4.7-flash-hf")

model = AutoModelForCausalLM.from_pretrained(
    "/opt/models/glm-4.7-flash-hf", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/opt/models/glm-4.7-flash-hf")
ids = tokenizer("Hello!", return_tensors="pt").input_ids.to(model.device)
print(tokenizer.decode(model.generate(ids, max_new_tokens=100)[0]))
EOF
```

When using the GreenBoost venv:

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    /opt/greenboost/venv/bin/python your_transformers_script.py
```

---

## ExLlamaV3

ExLlamaV3 loads CUDA through PyTorch:

```bash
greenboost run python your_exllama_script.py
```

When using the GreenBoost venv:

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    /opt/greenboost/venv/bin/python your_exllama_script.py
```

The GreenBoost-patched KV cache layer (`CacheLayer_greenboost`) allocates directly
from `/dev/greenboost` and does not go through the shim's cudaMalloc path for cache
tensors — `GREENBOOST_ACTIVE=1` is still needed for virtual VRAM reporting.

---

## Verify GreenBoost is active

```bash
# Should show T2 pool in use (non-zero) after loading a model larger than 12 GB:
cat /sys/class/greenboost/greenboost/pool_info

# Confirm virtual VRAM is visible (should report ~63 GB, not 12 GB):
greenboost run python -c "
import torch
print('VRAM reported:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

# Check shim is loaded into a running process (e.g. Ollama):
grep greenboost /proc/$(pgrep ollama)/maps | head -3
```

---
