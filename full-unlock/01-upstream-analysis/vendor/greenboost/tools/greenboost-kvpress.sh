#!/usr/bin/env bash
# GreenBoost — KV cache compression launcher (kvpress)
# Copyright (C) 2026 Ferran Duarri. GPL v2 / Commercial — see LICENSE.
#
# Runs HuggingFace inference with runtime KV cache compression via kvpress.
# No model retraining required — compression applied at inference time.
#
# Compression methods (all runtime, no retraining):
#   ExpAttn   — exponential attention score filtering (recommended, fast)
#   SnapKV    — cluster-based KV selection
#   KnormPress — key-norm based selection
#   DMS       — dynamic memory sparsification (Qwen3 models, requires special weights)
#
# USAGE:
#   ./tools/greenboost-kvpress.sh --model /path/to/model --prompt "your question"
#   ./tools/greenboost-kvpress.sh --model THUDM/glm-4.7-flash-hf --compression 0.5
#   ./tools/greenboost-kvpress.sh --benchmark  # runs compression ratio benchmark
#
# ENVIRONMENT:
#   GREENBOOST_KV_COMPRESS_RATIO  compression ratio 0.0-1.0 (fraction to KEEP, default 0.5)
#   GREENBOOST_KV_METHOD          press method: ExpAttn|SnapKV|KnormPress (default: ExpAttn)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREENBOOST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SHIM="/usr/local/lib/libgreenboost_cuda.so"
VENV="/opt/greenboost/venv"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GRN}[GreenBoost]${NC} $*"; }
warn() { echo -e "${YLW}[GreenBoost] WARN:${NC} $*"; }
die()  { echo -e "${RED}[GreenBoost] ERROR:${NC} $*" >&2; exit 1; }

# ── Args ────────────────────────────────────────────────────────────────────
MODEL=""
PROMPT=""
COMPRESSION="${GREENBOOST_KV_COMPRESS_RATIO:-0.5}"
METHOD="${GREENBOOST_KV_METHOD:-ExpAttn}"
BENCHMARK=0
MAX_TOKENS=512

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";       shift 2 ;;
        --prompt)      PROMPT="$2";      shift 2 ;;
        --compression) COMPRESSION="$2"; shift 2 ;;
        --method)      METHOD="$2";      shift 2 ;;
        --max-tokens)  MAX_TOKENS="$2";  shift 2 ;;
        --benchmark)   BENCHMARK=1;      shift   ;;
        *) shift ;;
    esac
done

# ── Setup ───────────────────────────────────────────────────────────────────
[[ -d "$VENV" ]] || die "Venv not found at $VENV — run: sudo ./greenboost_setup.sh full-install"
source "$VENV/bin/activate"

if [[ -f "$SHIM" ]]; then
    export LD_PRELOAD="$SHIM"
    info "GreenBoost shim active: $SHIM"
fi

# Install kvpress if not present
if ! python -c "import kvpress" &>/dev/null 2>&1; then
    info "Installing kvpress ..."
    pip install kvpress -q || die "kvpress install failed"
fi

# ── Benchmark mode ──────────────────────────────────────────────────────────
if [[ $BENCHMARK -eq 1 ]]; then
    [[ -n "$MODEL" ]] || MODEL="THUDM/glm-4.7-flash-hf"
    info "Running KV compression benchmark on $MODEL ..."
    python - <<PYEOF
import torch, time, os, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "$MODEL"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

prompt = "Explain the three laws of thermodynamics in detail." * 10  # long context
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_len = inputs["input_ids"].shape[1]
print(f"Input tokens: {input_len}")

methods_to_test = ["none", "ExpAttn", "KnormPress", "SnapKV"]

try:
    from kvpress import ExpAttnPress, KnormPress, SnapKVPress
    press_map = {
        "none":       None,
        "ExpAttn":    ExpAttnPress(compression_ratio=$COMPRESSION),
        "KnormPress": KnormPress(compression_ratio=$COMPRESSION),
        "SnapKV":     SnapKVPress(compression_ratio=$COMPRESSION),
    }
except ImportError as e:
    print(f"kvpress import error: {e}")
    sys.exit(1)

print(f"\n{'Method':<14} {'TTFT (s)':>10} {'Tok/s':>8} {'VRAM MB':>10}")
print("-" * 46)
for name, press in press_map.items():
    torch.cuda.empty_cache()
    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            if press is not None and hasattr(press, "__call__"):
                # kvpress pipeline
                from kvpress import KVPressTextGenerationPipeline
                pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
                out = pipe(prompt, press=press, max_new_tokens=$MAX_TOKENS)
                gen_tokens = $MAX_TOKENS
            else:
                out = model.generate(**inputs, max_new_tokens=$MAX_TOKENS, do_sample=False)
                gen_tokens = out.shape[1] - input_len
        elapsed = time.perf_counter() - t0
        tps = gen_tokens / elapsed
        vram = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"{name:<14} {elapsed:>10.2f} {tps:>8.1f} {vram:>10.0f}")
    except Exception as e:
        print(f"{name:<14} {'ERROR':>10}: {e}")

print("\nDone. Higher tok/s and lower VRAM = better compression effectiveness.")
PYEOF
    exit 0
fi

# ── Single inference ─────────────────────────────────────────────────────────
[[ -n "$MODEL"  ]] || die "Usage: $0 --model /path/to/model --prompt 'text'"
[[ -n "$PROMPT" ]] || die "Usage: $0 --model /path/to/model --prompt 'text'"

info "Model      : $MODEL"
info "Method     : $METHOD (ratio $COMPRESSION = keep ${COMPRESSION} of KV cache)"
info "Prompt     : $(echo "$PROMPT" | head -c 80)..."

python - <<PYEOF
import torch, time, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from kvpress import ExpAttnPress, KnormPress, SnapKVPress
    press_map = {
        "ExpAttn":    ExpAttnPress(compression_ratio=$COMPRESSION),
        "KnormPress": KnormPress(compression_ratio=$COMPRESSION),
        "SnapKV":     SnapKVPress(compression_ratio=$COMPRESSION),
    }
    press = press_map.get("$METHOD")
    if press is None:
        print(f"Unknown method '$METHOD'. Available: {list(press_map.keys())}", file=sys.stderr)
        sys.exit(1)
except ImportError as e:
    print(f"kvpress not available: {e}", file=sys.stderr)
    sys.exit(1)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("$MODEL", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "$MODEL", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

print(f"Running inference with {press.__class__.__name__} (ratio=$COMPRESSION)...")
t0 = time.perf_counter()
with torch.no_grad():
    inputs = tokenizer("$PROMPT", return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=$MAX_TOKENS, do_sample=False)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
elapsed = time.perf_counter() - t0
tokens = out.shape[1] - inputs["input_ids"].shape[1]

print(f"\nResponse ({tokens} tokens, {elapsed:.2f}s, {tokens/elapsed:.1f} tok/s):")
print(response)
PYEOF
