#!/usr/bin/env python3
"""
GreenBoost — Model Post-Training Quantization (PTQ) via NVIDIA ModelOpt.
Copyright (C) 2026 Ferran Duarri. GPL v2 / Commercial — see LICENSE.

Quantizes a HuggingFace model to FP8 (default), INT4-AWQ, or NVFP4 without
retraining. Reduces model size 2-4x, lowering GreenBoost Tier 2 System DDR pressure.

For glm-4.7-flash:q8_0 (32 GB):
  FP8 quantization → ~16 GB → fits in T1(12GB) + 4 GB T2 overflow
  INT4 quantization → ~8 GB  → fits entirely in T1 VRAM

Usage:
    python tools/greenboost-ptq.py --model THUDM/glm-4.7-flash-hf
    python tools/greenboost-ptq.py --model /path/to/model --quant fp8 --output /opt/models/out
    python tools/greenboost-ptq.py --model /path/to/model --quant int4_awq --calibration-samples 512

Output:
    /opt/greenboost/models/<model-name>-<quant>/   HuggingFace checkpoint
    /opt/greenboost/models/<model-name>-<quant>/Modelfile  Ollama Modelfile
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[GreenBoost PTQ] %(message)s")
log = logging.getLogger(__name__)

MODELOPT_DIR = Path(__file__).parent.parent / "libraries" / "Model-Optimizer"
VENV_PYTHON  = "/opt/greenboost/venv/bin/python"


def check_modelopt():
    """Verify ModelOpt is importable, install if not."""
    try:
        import modelopt
        log.info("ModelOpt version: %s", getattr(modelopt, "__version__", "unknown"))
        return True
    except ImportError:
        log.warning("ModelOpt not installed. Attempting install from %s ...", MODELOPT_DIR)
        if MODELOPT_DIR.exists():
            import subprocess
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(MODELOPT_DIR), "-q"],
                check=False
            )
            return r.returncode == 0
        log.error("ModelOpt directory not found at %s", MODELOPT_DIR)
        return False


def quantize(
    model_path: str,
    output_dir: str,
    quant: str = "fp8",
    calibration_samples: int = 512,
    dtype: str = "bfloat16",
):
    """Run PTQ quantization using ModelOpt."""
    import torch
    try:
        from modelopt.torch.quantization import quantize as mopt_quantize
        from modelopt.torch.quantization.config import PTQConfig
    except ImportError as e:
        log.error("ModelOpt import failed: %s", e)
        log.error("Install with: pip install -e %s", MODELOPT_DIR)
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model: %s", model_path)
    log.info("Quantization: %s | dtype: %s | calibration samples: %d",
             quant, dtype, calibration_samples)

    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Build calibration dataset
    log.info("Preparing calibration dataset (%d samples) ...", calibration_samples)
    calib_prompts = [
        "The capital of France is",
        "Quantum mechanics describes",
        "The history of artificial intelligence began",
        "In thermodynamics, entropy is",
    ] * (calibration_samples // 4 + 1)
    calib_prompts = calib_prompts[:calibration_samples]

    def calib_dataloader():
        for prompt in calib_prompts:
            tokens = tokenizer(prompt, return_tensors="pt")
            yield {k: v.to(model.device) for k, v in tokens.items()}

    # Map quant scheme to ModelOpt config
    quant_configs = {
        "fp8":      {"quant_dtype": "fp8",    "algorithm": "maxmin"},
        "int4_awq": {"quant_dtype": "int4",   "algorithm": "awq"},
        "nvfp4":    {"quant_dtype": "nvfp4",  "algorithm": "maxmin"},
        "int8":     {"quant_dtype": "int8",   "algorithm": "smoothquant"},
    }
    if quant not in quant_configs:
        log.error("Unknown quant scheme '%s'. Choose: %s", quant, list(quant_configs))
        sys.exit(1)

    cfg = PTQConfig(**quant_configs[quant])

    log.info("Running PTQ calibration ...")
    try:
        qmodel = mopt_quantize(model, config=cfg, forward_loop=calib_dataloader)
    except Exception as e:
        log.error("Quantization failed: %s", e)
        log.error("Tip: ensure CUDA is available and model fits in VRAM for calibration")
        sys.exit(1)

    # Save quantized checkpoint
    os.makedirs(output_dir, exist_ok=True)
    log.info("Saving quantized model to %s ...", output_dir)

    try:
        from modelopt.torch.export import export_hf_checkpoint
        export_hf_checkpoint(qmodel, export_dir=output_dir)
    except (ImportError, AttributeError):
        # Fallback: save via HF save_pretrained
        qmodel.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    meta = {
        "original_model": str(model_path),
        "quantization":   quant,
        "dtype":          dtype,
        "calibration_samples": calibration_samples,
        "greenboost_version": "2.3",
    }
    with open(os.path.join(output_dir, "greenboost_ptq_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Quantized model saved: %s", output_dir)
    return output_dir


def create_modelfile(output_dir: str, model_name: str, quant: str) -> str:
    """Generate an Ollama Modelfile for the quantized checkpoint."""
    modelfile_path = os.path.join(output_dir, "Modelfile")
    content = f"""# GreenBoost PTQ Modelfile — {model_name} ({quant})
# Generated by greenboost-ptq.py
# Author: Ferran Duarri

FROM {output_dir}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 131072

SYSTEM You are a helpful AI assistant.
"""
    with open(modelfile_path, "w") as f:
        f.write(content)

    log.info("Ollama Modelfile created: %s", modelfile_path)
    log.info("")
    log.info("To load in Ollama:")
    log.info("  ollama create %s-%s -f %s", model_name, quant, modelfile_path)
    log.info("  ollama run %s-%s", model_name, quant)
    return modelfile_path


def main():
    parser = argparse.ArgumentParser(
        description="GreenBoost PTQ — quantize HF model for faster local inference"
    )
    parser.add_argument("--model",    required=True, help="HF model name or local path")
    parser.add_argument("--output",   default=None,  help="Output directory (default: auto)")
    parser.add_argument("--quant",    default="fp8",
                        choices=["fp8", "int4_awq", "nvfp4", "int8"],
                        help="Quantization scheme (default: fp8)")
    parser.add_argument("--dtype",    default="bfloat16",
                        help="Base dtype (default: bfloat16)")
    parser.add_argument("--calibration-samples", type=int, default=512,
                        help="Calibration samples for PTQ (default: 512)")
    parser.add_argument("--create-modelfile", action="store_true",
                        help="Generate Ollama Modelfile after quantization")
    args = parser.parse_args()

    # Determine output path
    model_name = os.path.basename(args.model.rstrip("/"))
    output_dir = args.output or f"/opt/greenboost/models/{model_name}-{args.quant}"

    log.info("GreenBoost Model PTQ")
    log.info("  Model   : %s", args.model)
    log.info("  Quant   : %s", args.quant)
    log.info("  Output  : %s", output_dir)
    log.info("  Calib   : %d samples", args.calibration_samples)

    # Verify/install ModelOpt
    if not check_modelopt():
        log.error("ModelOpt unavailable. Install: pip install nvidia-modelopt[all]")
        log.error("or from source: pip install -e %s", MODELOPT_DIR)
        sys.exit(1)

    # Run quantization
    quantize(
        model_path=args.model,
        output_dir=output_dir,
        quant=args.quant,
        calibration_samples=args.calibration_samples,
        dtype=args.dtype,
    )

    # Create Modelfile
    if args.create_modelfile:
        create_modelfile(output_dir, model_name, args.quant)

    log.info("")
    log.info("Quantization complete!")
    log.info("Expected size reduction: %s → %s",
             {"fp8": "2×", "int4_awq": "4×", "nvfp4": "4×", "int8": "2×"}[args.quant],
             "less GreenBoost Tier 2 pressure = faster inference")


if __name__ == "__main__":
    main()
