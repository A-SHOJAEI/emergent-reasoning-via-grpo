#!/usr/bin/env python3
"""Evaluate base, SFT, and GRPO models on GSM8K and MATH benchmarks."""

import sys
sys.path.insert(0, ".")

import argparse
import json
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.grpo_reasoning.data.dataset import load_gsm8k, load_math
from src.grpo_reasoning.evaluation.metrics import (
    analyze_reasoning_patterns,
    evaluate_accuracy,
    evaluate_pass_at_k,
)
from src.grpo_reasoning.utils.config import ensure_dirs, load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    base_model_name: str,
    adapter_path: str | None = None,
):
    """Load base model with optional LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def evaluate_model(
    model,
    tokenizer,
    eval_samples: list[dict],
    model_label: str,
    config: dict,
    results_dir: Path,
):
    """Run full evaluation for a single model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_label}")
    logger.info(f"{'='*60}")

    eval_cfg = config["evaluation"]

    # Greedy accuracy
    logger.info("Computing greedy accuracy...")
    acc_results = evaluate_accuracy(
        model, tokenizer, eval_samples,
        max_new_tokens=eval_cfg["max_new_tokens"],
        temperature=eval_cfg["temperature"],
    )
    logger.info(f"  Accuracy: {acc_results['accuracy']:.4f} "
                f"({acc_results['correct']}/{acc_results['total']})")
    for src, acc in acc_results["per_source_accuracy"].items():
        count = acc_results["per_source_counts"][src]
        logger.info(f"  {src}: {acc:.4f} ({count['correct']}/{count['total']})")

    # Reasoning pattern analysis
    logger.info("Analyzing reasoning patterns...")
    patterns = analyze_reasoning_patterns(acc_results["predictions"])
    logger.info(f"  Step numbers:    {patterns['uses_step_numbers']:.1%}")
    logger.info(f"  Equations:       {patterns['uses_equations']:.1%}")
    logger.info(f"  Therefore/thus:  {patterns['uses_therefore']:.1%}")
    logger.info(f"  #### delimiter:  {patterns['uses_hashtag_delimiter']:.1%}")
    logger.info(f"  Avg length:      {patterns['avg_response_length']:.0f} chars")
    logger.info(f"  Avg lines:       {patterns['avg_num_lines']:.1f}")

    # Save results
    output = {
        "model": model_label,
        "accuracy": acc_results["accuracy"],
        "correct": acc_results["correct"],
        "total": acc_results["total"],
        "per_source_accuracy": acc_results["per_source_accuracy"],
        "per_source_counts": acc_results["per_source_counts"],
        "reasoning_patterns": patterns,
        "sample_predictions": acc_results["predictions"][:20],
    }

    output_path = results_dir / f"{model_label.lower().replace(' ', '_')}_eval.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning models")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--grpo-adapter", type=str, default=None,
                        help="Path to GRPO LoRA adapter")
    parser.add_argument("--sft-adapter", type=str, default=None,
                        help="Path to SFT LoRA adapter")
    parser.add_argument("--eval-base", action="store_true",
                        help="Also evaluate the base model (no adapter)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override max eval samples")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["grpo"]["seed"])
    ensure_dirs(config)

    results_dir = Path(config["paths"]["results_dir"])
    max_samples = args.max_samples or config["data"]["max_eval_samples"]
    base_model_name = config["model"]["name"]

    # ── Load eval data ───────────────────────────────────────────
    logger.info("Loading evaluation data...")
    eval_samples = []

    gsm8k_test = load_gsm8k("test", max_samples=max_samples)
    eval_samples.extend(gsm8k_test)
    logger.info(f"  GSM8K test: {len(gsm8k_test)} samples")

    math_test = load_math("test", max_samples=max_samples)
    eval_samples.extend(math_test)
    logger.info(f"  MATH test: {len(math_test)} samples")

    logger.info(f"  Total eval samples: {len(eval_samples)}")

    all_results = {}

    # ── Evaluate base model ──────────────────────────────────────
    if args.eval_base:
        model, tokenizer = load_model_and_tokenizer(base_model_name)
        result = evaluate_model(
            model, tokenizer, eval_samples, "Base", config, results_dir,
        )
        all_results["base"] = result
        del model
        torch.cuda.empty_cache()

    # ── Evaluate SFT model ───────────────────────────────────────
    if args.sft_adapter:
        model, tokenizer = load_model_and_tokenizer(
            base_model_name, args.sft_adapter,
        )
        result = evaluate_model(
            model, tokenizer, eval_samples, "SFT", config, results_dir,
        )
        all_results["sft"] = result
        del model
        torch.cuda.empty_cache()

    # ── Evaluate GRPO model ──────────────────────────────────────
    if args.grpo_adapter:
        model, tokenizer = load_model_and_tokenizer(
            base_model_name, args.grpo_adapter,
        )
        result = evaluate_model(
            model, tokenizer, eval_samples, "GRPO", config, results_dir,
        )
        all_results["grpo"] = result
        del model
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    summary = {}
    for label, result in all_results.items():
        logger.info(f"\n{result['model']}:")
        logger.info(f"  Overall accuracy: {result['accuracy']:.4f}")
        for src, acc in result["per_source_accuracy"].items():
            logger.info(f"  {src}: {acc:.4f}")
        summary[label] = {
            "accuracy": result["accuracy"],
            "per_source": result["per_source_accuracy"],
            "reasoning_patterns": result["reasoning_patterns"],
        }

    summary_path = results_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
