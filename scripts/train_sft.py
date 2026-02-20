#!/usr/bin/env python3
"""Train SFT (Supervised Fine-Tuning) baseline for comparison with GRPO."""

import sys
sys.path.insert(0, ".")

import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from src.grpo_reasoning.data.dataset import build_sft_dataset, load_gsm8k
from src.grpo_reasoning.utils.config import ensure_dirs, load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = load_config("configs/default.yaml")
    set_seed(config["grpo"]["seed"])
    ensure_dirs(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["paths"]["output_dir"]) / f"sft_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SFT Baseline Training")
    logger.info("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    logger.info("Loading GSM8K training data for SFT...")
    train_samples = load_gsm8k("train", max_samples=config["data"]["max_train_samples"])
    sft_data = build_sft_dataset(train_samples)
    train_dataset = Dataset.from_list(sft_data)
    logger.info(f"  SFT dataset size: {len(train_dataset)}")

    # ── Load model ───────────────────────────────────────────────
    model_name = config["model"]["name"]
    logger.info(f"Loading model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA config ──────────────────────────────────────────────
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"],
    )

    # ── SFT config ───────────────────────────────────────────────
    sft_cfg = config["sft"]

    training_args = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=sft_cfg["num_train_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=sft_cfg["warmup_ratio"],
        weight_decay=sft_cfg["weight_decay"],
        max_length=sft_cfg["max_seq_length"],
        bf16=sft_cfg["bf16"],
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        seed=config["grpo"]["seed"],
        report_to="tensorboard",
        logging_dir=str(Path(config["paths"]["logs_dir"]) / f"sft_{timestamp}"),
    )

    # ── Trainer ──────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info(f"  Trainable parameters (with LoRA): {trainable:,}")

    # ── Train ────────────────────────────────────────────────────
    logger.info("Starting SFT training...")
    train_result = trainer.train()

    # ── Save ─────────────────────────────────────────────────────
    trainer.save_model(str(run_dir / "final"))
    tokenizer.save_pretrained(str(run_dir / "final"))

    metrics = train_result.metrics
    with open(Path(config["paths"]["results_dir"]) / "sft_training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info("SFT Training complete!")
    logger.info(f"  Output: {run_dir}")
    logger.info(f"  Total steps: {train_result.global_step}")
    logger.info(f"  Final loss: {metrics.get('train_loss', 'N/A')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
