#!/usr/bin/env python3
"""
GreenBoost — LoRA fine-tuning using Unsloth (fastest LoRA trainer on NVIDIA GPUs).
Copyright (C) 2026 Ferran Duarri. GPL v2 / Commercial — see LICENSE.

Fine-tunes a model on custom data using LoRA (r=16). Fits in 12 GB VRAM via
4-bit base model. Exports merged GGUF for Ollama or standalone HF checkpoint.

VRAM breakdown for RTX 5070 (12 GB):
  Base model 4-bit (GLM-4.7-Flash 30B): ~8 GB
  LoRA parameters (r=16): ~0.3 GB
  Activations + optimizer states: ~3 GB
  Total: ~11.3 GB — fits in 12 GB with headroom

Usage:
    python tools/greenboost-lora-train.py \
        --model THUDM/glm-4.7-flash-hf \
        --data /path/to/data.jsonl \
        --output /opt/greenboost/models/glm-lora \
        --epochs 3

Data format (JSONL, one example per line):
    {"instruction": "...", "input": "...", "output": "..."}
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

After training, merge and export:
    python tools/greenboost-lora-train.py --merge-only --lora /opt/greenboost/models/glm-lora
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[GreenBoost LoRA] %(message)s")
log = logging.getLogger(__name__)

LORA_DIR = Path(__file__).parent.parent / "libraries" / "LoRA"


def check_unsloth():
    try:
        import unsloth
        log.info("Unsloth version: %s", getattr(unsloth, "__version__", "unknown"))
        return True
    except ImportError:
        log.warning("Unsloth not installed. Install with:")
        log.warning("  pip install unsloth")
        log.warning("  # or (editable from loralib source):")
        log.warning("  pip install -e %s", LORA_DIR)
        return False


def load_dataset(data_path: str):
    """Load JSONL dataset in instruction or messages format."""
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj:
                examples.append(obj)
            elif "instruction" in obj:
                # Convert instruction format to messages
                content = obj["instruction"]
                if obj.get("input"):
                    content += "\n" + obj["input"]
                examples.append({
                    "messages": [
                        {"role": "user",      "content": content},
                        {"role": "assistant", "content": obj.get("output", "")},
                    ]
                })
    log.info("Loaded %d training examples from %s", len(examples), data_path)
    return examples


def train(
    model_path: str,
    data_path: str,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    batch_size: int = 2,
    max_seq_len: int = 4096,
    learning_rate: float = 2e-4,
):
    """Run LoRA fine-tuning with Unsloth."""
    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError:
        log.error("Unsloth required: pip install unsloth")
        sys.exit(1)

    log.info("Loading base model (4-bit) for LoRA training ...")
    log.info("  Model: %s | rank r=%d | max_seq_len=%d", model_path, lora_r, max_seq_len)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_len,
        dtype=None,           # auto-detect: bfloat16 on RTX 5070
        load_in_4bit=True,    # QLoRA: 4-bit base for VRAM efficiency
    )

    # Inject LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.0,     # 0 is optimal per Unsloth benchmarks
        bias="none",
        use_gradient_checkpointing="unsloth",  # VRAM-efficient checkpointing
        random_state=42,
    )

    log.info("LoRA parameters: r=%d, alpha=%d", lora_r, lora_alpha)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable: %d / %d params (%.2f%%)",
             trainable_params, total_params, 100 * trainable_params / total_params)

    # Load data
    examples = load_dataset(data_path)

    from datasets import Dataset
    ds = Dataset.from_list(examples)

    def format_example(ex):
        if "messages" in ex:
            # Chat template
            return {"text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )}
        return {"text": ex.get("text", "")}

    ds = ds.map(format_example, remove_columns=ds.column_names)

    # Training
    try:
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        log.error("trl required: pip install trl")
        sys.exit(1)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="epoch",
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
        ),
    )

    log.info("Starting training (%d epochs, batch=%d) ...", epochs, batch_size)
    trainer.train()
    log.info("Training complete.")

    # Save LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("LoRA adapter saved: %s", output_dir)

    return model, tokenizer


def merge_and_export(lora_dir: str, export_dir: str, export_gguf: bool = True):
    """Merge LoRA weights into base model and optionally export GGUF for Ollama."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        log.error("Unsloth required: pip install unsloth")
        sys.exit(1)

    log.info("Loading LoRA adapter from %s ...", lora_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        lora_dir, max_seq_length=4096, dtype=None, load_in_4bit=True
    )

    # Merge LoRA weights into base model
    merged_dir = os.path.join(export_dir, "merged")
    log.info("Merging LoRA into base model → %s ...", merged_dir)
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    log.info("Merged model saved: %s", merged_dir)

    if export_gguf:
        gguf_dir = os.path.join(export_dir, "gguf")
        os.makedirs(gguf_dir, exist_ok=True)
        log.info("Exporting GGUF for Ollama → %s ...", gguf_dir)
        # Q8 GGUF for highest quality
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q8_0")
        log.info("GGUF exported: %s", gguf_dir)

        # Create Ollama Modelfile
        modelfile_path = os.path.join(gguf_dir, "Modelfile")
        gguf_files = list(Path(gguf_dir).glob("*.gguf"))
        if gguf_files:
            gguf_path = gguf_files[0]
            with open(modelfile_path, "w") as f:
                f.write(f"""# GreenBoost LoRA merged model
FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER num_ctx 32768
SYSTEM You are a helpful AI assistant (LoRA fine-tuned via GreenBoost).
""")
            log.info("")
            log.info("Ollama Modelfile: %s", modelfile_path)
            log.info("Load in Ollama:")
            log.info("  ollama create my-model -f %s", modelfile_path)
            log.info("  ollama run my-model")


def main():
    parser = argparse.ArgumentParser(
        description="GreenBoost LoRA fine-tuning (Unsloth, fits 30B in 12 GB VRAM)"
    )
    parser.add_argument("--model",  default=None, help="Base model (HF name or path)")
    parser.add_argument("--data",   default=None, help="Training data JSONL path")
    parser.add_argument("--output", default="/opt/greenboost/models/lora-adapter",
                        help="Output directory for LoRA adapter")
    parser.add_argument("--epochs",   type=int,   default=3)
    parser.add_argument("--batch",    type=int,   default=2)
    parser.add_argument("--rank",     type=int,   default=16, help="LoRA rank r")
    parser.add_argument("--max-len",  type=int,   default=4096)
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip training, only merge existing LoRA adapter")
    parser.add_argument("--lora",   default=None,
                        help="Path to existing LoRA adapter (for --merge-only)")
    parser.add_argument("--no-gguf", action="store_true",
                        help="Skip GGUF export after merge")
    args = parser.parse_args()

    if not check_unsloth():
        sys.exit(1)

    if args.merge_only:
        lora_dir = args.lora or args.output
        export_dir = args.output
        if not os.path.exists(lora_dir):
            log.error("LoRA directory not found: %s", lora_dir)
            sys.exit(1)
        merge_and_export(lora_dir, export_dir, export_gguf=not args.no_gguf)
        return

    if not args.model:
        log.error("--model required for training")
        sys.exit(1)
    if not args.data:
        log.error("--data required for training (JSONL file)")
        sys.exit(1)

    model, tokenizer = train(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        lora_r=args.rank,
        epochs=args.epochs,
        batch_size=args.batch,
        max_seq_len=args.max_len,
    )

    # Optionally merge after training
    merge_and_export(args.output, args.output, export_gguf=not args.no_gguf)


if __name__ == "__main__":
    main()
