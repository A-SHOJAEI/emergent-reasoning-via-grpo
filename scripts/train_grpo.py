#!/usr/bin/env python3
"""Train a language model with GRPO (Group Relative Policy Optimization).

Reproduces the DeepSeek-R1 methodology: pure reinforcement learning
produces emergent chain-of-thought reasoning without supervised traces.
"""

import sys
sys.path.insert(0, ".")

import json
import logging
import os
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
from trl import GRPOConfig, GRPOTrainer

from src.grpo_reasoning.data.dataset import build_grpo_dataset, load_gsm8k
from src.grpo_reasoning.models.reward import build_reward_fn
from src.grpo_reasoning.utils.config import ensure_dirs, load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    config = load_config("configs/default.yaml")
    set_seed(config["grpo"]["seed"])
    ensure_dirs(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["paths"]["output_dir"]) / f"grpo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GRPO Training — Emergent Reasoning via Reinforcement Learning")
    logger.info("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    logger.info("Loading GSM8K training data...")
    train_samples = load_gsm8k("train", max_samples=config["data"]["max_train_samples"])
    logger.info(f"  Train samples: {len(train_samples)}")

    grpo_data = build_grpo_dataset(train_samples)
    train_dataset = Dataset.from_list(grpo_data)

    logger.info(f"  GRPO dataset size: {len(train_dataset)}")

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
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {param_count:,}")
    logger.info(f"  Trainable parameters (before LoRA): {trainable_count:,}")

    # ── LoRA config ──────────────────────────────────────────────
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"],
    )

    # ── Reward function ──────────────────────────────────────────
    reward_fn = build_reward_fn(config)
    logger.info("Reward function: correctness + format bonus")

    # ── GRPO config ──────────────────────────────────────────────
    grpo_cfg = config["grpo"]

    grpo_kwargs = dict(
        output_dir=str(run_dir),
        num_train_epochs=grpo_cfg["num_train_epochs"],
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_cfg["gradient_accumulation_steps"],
        learning_rate=grpo_cfg["learning_rate"],
        lr_scheduler_type=grpo_cfg["lr_scheduler_type"],
        warmup_ratio=grpo_cfg["warmup_ratio"],
        weight_decay=grpo_cfg["weight_decay"],
        max_grad_norm=grpo_cfg["max_grad_norm"],
        bf16=grpo_cfg["bf16"],
        logging_steps=grpo_cfg["logging_steps"],
        save_steps=grpo_cfg["save_steps"],
        save_total_limit=3,
        seed=grpo_cfg["seed"],
        num_generations=grpo_cfg["num_generations"],
        max_completion_length=grpo_cfg["max_completion_length"],
        temperature=grpo_cfg["temperature"],
        beta=grpo_cfg["beta"],
        epsilon=grpo_cfg["epsilon"],
        loss_type=grpo_cfg["loss_type"],
        num_iterations=grpo_cfg["num_iterations"],
        report_to="tensorboard",
        logging_dir=str(Path(config["paths"]["logs_dir"]) / f"grpo_{timestamp}"),
        remove_unused_columns=False,
        log_completions=True,
    )
    if grpo_cfg.get("max_steps"):
        grpo_kwargs["max_steps"] = grpo_cfg["max_steps"]

    training_args = GRPOConfig(**grpo_kwargs)

    # ── Trainer ──────────────────────────────────────────────────
    logger.info("Initializing GRPOTrainer...")

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainable_after = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    logger.info(f"  Trainable parameters (with LoRA): {trainable_after:,}")

    # ── Train ────────────────────────────────────────────────────
    logger.info("Starting GRPO training...")
    logger.info(f"  Epochs: {grpo_cfg['num_train_epochs']}")
    logger.info(f"  Batch size: {grpo_cfg['per_device_train_batch_size']}")
    logger.info(f"  Grad accum: {grpo_cfg['gradient_accumulation_steps']}")
    logger.info(f"  Group size (G): {grpo_cfg['num_generations']}")
    logger.info(f"  Max completion length: {grpo_cfg['max_completion_length']}")
    logger.info(f"  Learning rate: {grpo_cfg['learning_rate']}")

    train_result = trainer.train()

    # ── Save ─────────────────────────────────────────────────────
    logger.info("Saving model and tokenizer...")
    trainer.save_model(str(run_dir / "final"))
    tokenizer.save_pretrained(str(run_dir / "final"))

    # Save training stats
    metrics = train_result.metrics
    metrics_path = Path(config["paths"]["results_dir"]) / "grpo_training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")

    # Save config used
    with open(run_dir / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)

    logger.info("=" * 60)
    logger.info("GRPO Training complete!")
    logger.info(f"  Output: {run_dir}")
    logger.info(f"  Total steps: {train_result.global_step}")
    logger.info(f"  Final loss: {metrics.get('train_loss', 'N/A')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
