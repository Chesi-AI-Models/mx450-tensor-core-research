#!/usr/bin/env bash
# GreenBoost — ExLlamaV3 launcher
# Copyright (C) 2026 Ferran Duarri. GPL v2 / Commercial — see LICENSE.
#
# Runs ExLlamaV3 chat or server with GreenBoost System DDR KV cache offload enabled.
# Automatically injects LD_PRELOAD shim and sets GreenBoost cache environment.
#
# USAGE:
#   ./tools/greenboost-exllama.sh --model /path/to/model [--mode chat|server] [options]
#
# EXAMPLES:
#   # Interactive chat with GLM-4.7-Flash EXL3 4bpw:
#   ./tools/greenboost-exllama.sh --model /opt/models/glm-4.7-flash-exl3
#
#   # OpenAI-compatible API server on port 8080:
#   ./tools/greenboost-exllama.sh --model /opt/models/glm-4.7-flash-exl3 --mode server --port 8080
#
#   # With EXL3 conversion from HF model (slow first run):
#   ./tools/greenboost-exllama.sh --model THUDM/glm-4.7-flash-hf --exl3-convert --bpw 4.0
#
# ENVIRONMENT:
#   GREENBOOST_DEBUG=1            verbose shim logging
#   GREENBOOST_CACHE_VERBOSE=1    verbose cache allocation logging
#   GREENBOOST_KV_CTX             max context tokens (default: 32768)
#   EXLLAMA_VENV                  path to venv (default: /opt/greenboost/venv)
#   MODEL_DIR                     default model directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREENBOOST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SHIM="/usr/local/lib/libgreenboost_cuda.so"
VENV="${EXLLAMA_VENV:-/opt/greenboost/venv}"
EXLLAMA_DIR="${GREENBOOST_DIR}/libraries/exllamav3"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; BLU='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${GRN}[GreenBoost]${NC} $*"; }
warn()  { echo -e "${YLW}[GreenBoost] WARN:${NC} $*"; }
die()   { echo -e "${RED}[GreenBoost] ERROR:${NC} $*" >&2; exit 1; }

# ── Argument parsing ────────────────────────────────────────────────────────
MODEL=""
MODE="chat"
PORT=8080
CTX="${GREENBOOST_KV_CTX:-32768}"
BPW="4.0"
EXL3_CONVERT=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";      shift 2 ;;
        --mode)        MODE="$2";       shift 2 ;;
        --port)        PORT="$2";       shift 2 ;;
        --ctx)         CTX="$2";        shift 2 ;;
        --bpw)         BPW="$2";        shift 2 ;;
        --exl3-convert) EXL3_CONVERT=1; shift   ;;
        *)             EXTRA_ARGS+=("$1"); shift ;;
    esac
done

[[ -z "$MODEL" ]] && die "Usage: $0 --model /path/to/model [--mode chat|server] [--ctx 32768]"

# ── Validate environment ────────────────────────────────────────────────────
info "GreenBoost ExLlamaV3 launcher"
info "  Model : $MODEL"
info "  Mode  : $MODE"
info "  CTX   : $CTX tokens"

# Check GreenBoost module
if ! lsmod | grep -q "^greenboost "; then
    warn "greenboost.ko not loaded — KV cache will fall back to CPU RAM"
    warn "Load with: sudo ./greenboost_setup.sh load"
fi

# Check shim
if [[ -f "$SHIM" ]]; then
    info "  Shim  : $SHIM ✓"
    export LD_PRELOAD="$SHIM"
else
    warn "  Shim $SHIM not installed — CUDA alloc not intercepted"
    warn "  Install with: sudo ./greenboost_setup.sh install"
fi

# Activate venv
if [[ -d "$VENV" ]]; then
    source "$VENV/bin/activate"
    info "  Venv  : $VENV ✓"
else
    die "Python venv not found at $VENV — run: sudo ./greenboost_setup.sh full-install"
fi

# Check ExLlamaV3
if ! python -c "import exllamav3" &>/dev/null 2>&1; then
    info "Installing ExLlamaV3 from $EXLLAMA_DIR ..."
    pip install -e "$EXLLAMA_DIR" --no-build-isolation -q \
        || die "ExLlamaV3 install failed — see output above"
fi

# ── EXL3 conversion (optional) ─────────────────────────────────────────────
if [[ $EXL3_CONVERT -eq 1 ]]; then
    local EXL3_OUT="/opt/greenboost/models/$(basename "$MODEL")-exl3-${BPW}bpw"
    local WORK_DIR="/tmp/greenboost-exl3-work"
    info "Converting $MODEL → EXL3 @ ${BPW} bpw ..."
    info "Output: $EXL3_OUT"
    mkdir -p "$EXL3_OUT" "$WORK_DIR"
    python "$EXLLAMA_DIR/convert.py" \
        -i "$MODEL" \
        -o "$EXL3_OUT" \
        -w "$WORK_DIR" \
        -b "$BPW" \
        || die "EXL3 conversion failed"
    info "Conversion complete: $EXL3_OUT"
    MODEL="$EXL3_OUT"
fi

# ── GreenBoost cache environment ────────────────────────────────────────────
export GREENBOOST_CACHE_VERBOSE="${GREENBOOST_DEBUG:-0}"
export PYTHONPATH="$EXLLAMA_DIR:${PYTHONPATH:-}"

# ── Launch ──────────────────────────────────────────────────────────────────
echo ""
info "Starting ExLlamaV3 ($MODE) — KV cache → GreenBoost Tier 2 System DDR"
echo ""

case "$MODE" in
    chat)
        exec python "$EXLLAMA_DIR/examples/chat.py" \
            -m "$MODEL" \
            -cache_mode greenboost \
            "${EXTRA_ARGS[@]}"
        ;;
    server)
        # TabbyAPI-style OpenAI-compatible server
        info "API server: http://localhost:${PORT}/v1"
        exec python -m exllamav3.server \
            --model "$MODEL" \
            --host 127.0.0.1 \
            --port "$PORT" \
            --max-seq-len "$CTX" \
            --cache-mode greenboost \
            "${EXTRA_ARGS[@]}" 2>/dev/null \
            || exec python "$EXLLAMA_DIR/examples/chat_console.py" \
                -m "$MODEL" \
                -cache_mode greenboost \
                "${EXTRA_ARGS[@]}"
        ;;
    *)
        die "Unknown mode '$MODE' — use: chat | server"
        ;;
esac
