#!/usr/bin/env bash
# GreenBoost — Convert HuggingFace model to EXL3 format for ExLlamaV3
# Copyright (C) 2026 Ferran Duarri. GPL v2 / Commercial — see LICENSE.
#
# EXL3 quantization achieves 2–8 bpw (bits per weight). For glm-4.7-flash:
#   8 bpw → ~32 GB (same as q8_0; not useful)
#   4 bpw → ~16 GB (fits in T1 + 4 GB T2 overflow)
#   2 bpw → ~8 GB  (fits entirely in T1 VRAM — maximum speed)
#
# Recommended: 4.0 bpw for best quality/speed balance.
#
# USAGE:
#   ./tools/convert_to_exl3.sh --model /path/to/hf-model
#   ./tools/convert_to_exl3.sh --model THUDM/glm-4.7-flash-hf --bpw 4.0
#   ./tools/convert_to_exl3.sh --model /path/to/model --bpw 2.0 --output /opt/models/glm-exl3-2bpw
#
# OUTPUT:
#   /opt/greenboost/models/<model-name>-exl3-<bpw>bpw/
#
# NOTES:
#   - Requires ~48 GB RAM + model size free disk during conversion
#   - Conversion is CPU-bound, takes 20–90 min for a 30B model
#   - Output can be loaded directly by tools/greenboost-exllama.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREENBOOST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXLLAMA_DIR="$GREENBOOST_DIR/libraries/exllamav3"
VENV="${EXLLAMA_VENV:-/opt/greenboost/venv}"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GRN}[GreenBoost EXL3]${NC} $*"; }
warn() { echo -e "${YLW}[GreenBoost EXL3] WARN:${NC} $*"; }
die()  { echo -e "${RED}[GreenBoost EXL3] ERROR:${NC} $*" >&2; exit 1; }

# ── Argument parsing ─────────────────────────────────────────────────────────
MODEL=""
BPW="4.0"
OUTPUT=""
CALIBRATION_ROWS=128
WORK_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";            shift 2 ;;
        --bpw)        BPW="$2";              shift 2 ;;
        --output)     OUTPUT="$2";           shift 2 ;;
        --calib-rows) CALIBRATION_ROWS="$2"; shift 2 ;;
        --work-dir)   WORK_DIR="$2";         shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) die "Unknown argument: $1  (use --help)" ;;
    esac
done

[[ -z "$MODEL" ]] && die "Usage: $0 --model /path/to/model [--bpw 4.0] [--output /path/to/out]"

# ── Resolve paths ────────────────────────────────────────────────────────────
MODEL_NAME="$(basename "${MODEL%/}")"
OUTPUT="${OUTPUT:-/opt/greenboost/models/${MODEL_NAME}-exl3-${BPW}bpw}"
WORK_DIR="${WORK_DIR:-/tmp/greenboost-exl3-work-$$}"

info "═══════════════════════════════════════════════════════════════"
info "  GreenBoost — EXL3 Conversion"
info "  Copyright (C) 2026 Ferran Duarri"
info "═══════════════════════════════════════════════════════════════"
info "  Model  : $MODEL"
info "  BPW    : $BPW bits/weight"
info "  Output : $OUTPUT"
info "  Work   : $WORK_DIR"
info "═══════════════════════════════════════════════════════════════"
echo ""

# ── Size estimate ─────────────────────────────────────────────────────────────
warn "Size estimate for 30B model:"
warn "  4.0 bpw → ~16 GB output | 2.0 bpw → ~8 GB | 8.0 bpw → ~32 GB"
warn "Conversion takes 20-90 min on i9-14900KF. Don't interrupt."
echo ""

# ── Check ExLlamaV3 ───────────────────────────────────────────────────────────
[[ -f "$EXLLAMA_DIR/convert.py" ]] || die "ExLlamaV3 convert.py not found at $EXLLAMA_DIR/convert.py"
[[ -d "$VENV" ]] || die "Python venv not found at $VENV — run: sudo ./greenboost_setup.sh full-install"

source "$VENV/bin/activate"

if ! python -c "import exllamav3" &>/dev/null 2>&1; then
    info "Installing ExLlamaV3 from $EXLLAMA_DIR ..."
    pip install -e "$EXLLAMA_DIR" --no-build-isolation -q \
        || die "ExLlamaV3 install failed"
fi

# ── Create output + work dirs ─────────────────────────────────────────────────
mkdir -p "$OUTPUT" "$WORK_DIR"

# ── Run conversion ─────────────────────────────────────────────────────────────
info "Starting EXL3 conversion (bpw=$BPW, calib_rows=$CALIBRATION_ROWS) ..."
echo ""

python "$EXLLAMA_DIR/convert.py" \
    -i "$MODEL" \
    -o "$OUTPUT" \
    -w "$WORK_DIR" \
    -b "$BPW" \
    -r "$CALIBRATION_ROWS" \
    || die "EXL3 conversion failed — check output above"

# Clean up temp work dir
rm -rf "$WORK_DIR"

info ""
info "Conversion complete!"
info "Output : $OUTPUT"
info ""
info "Load with ExLlamaV3:"
info "  ./tools/greenboost-exllama.sh --model $OUTPUT"
info ""
info "Or run GreenBoost optimize-model:"
info "  sudo ./greenboost_setup.sh optimize-model --model $OUTPUT --strategy exllama"
