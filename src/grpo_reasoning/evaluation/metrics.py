"""Evaluation metrics for math reasoning."""

import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.grpo_reasoning.data.dataset import (
    SYSTEM_PROMPT,
    extract_model_answer,
    normalize_answer,
)


def evaluate_accuracy(
    model,
    tokenizer,
    samples: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    batch_size: int = 4,
) -> dict:
    """Evaluate model accuracy on a set of math problems.

    Returns dict with accuracy, per-source accuracy, and sample predictions.
    """
    model.eval()
    correct = 0
    total = 0
    results_by_source = {}
    predictions = []

    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch = samples[i : i + batch_size]

        for sample in batch:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": sample["prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                }
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature

                output_ids = model.generate(**inputs, **gen_kwargs)

            # Decode only the generated part
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            model_ans = normalize_answer(extract_model_answer(generated))
            expected = normalize_answer(sample["answer"])
            is_correct = model_ans == expected

            source = sample.get("source", "unknown")
            if source not in results_by_source:
                results_by_source[source] = {"correct": 0, "total": 0}
            results_by_source[source]["total"] += 1

            if is_correct:
                correct += 1
                results_by_source[source]["correct"] += 1

            total += 1

            predictions.append({
                "prompt": sample["prompt"],
                "expected": expected,
                "predicted": model_ans,
                "correct": is_correct,
                "generated_text": generated[:500],
                "source": source,
            })

    accuracy = correct / total if total > 0 else 0.0
    per_source = {
        src: data["correct"] / data["total"] if data["total"] > 0 else 0.0
        for src, data in results_by_source.items()
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_source_accuracy": per_source,
        "per_source_counts": results_by_source,
        "predictions": predictions,
    }


def evaluate_pass_at_k(
    model,
    tokenizer,
    samples: list[dict],
    k_values: list[int] = [1, 5],
    num_samples: int = 10,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> dict:
    """Evaluate pass@k: probability that at least one of k samples is correct."""
    model.eval()
    pass_at_k = {k: 0 for k in k_values}
    total = 0

    for sample in tqdm(samples, desc="Pass@k evaluation"):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate multiple completions
        correct_count = 0
        with torch.no_grad():
            for _ in range(num_samples):
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                generated = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                model_ans = normalize_answer(extract_model_answer(generated))
                expected = normalize_answer(sample["answer"])
                if model_ans == expected:
                    correct_count += 1

        # Compute pass@k for each k
        for k in k_values:
            if correct_count >= 1:
                # pass@k = 1 - C(n-c, k) / C(n, k)
                n = num_samples
                c = correct_count
                if n - c < k:
                    pass_at_k[k] += 1.0
                else:
                    prob_fail = 1.0
                    for i in range(k):
                        prob_fail *= (n - c - i) / (n - i)
                    pass_at_k[k] += 1.0 - prob_fail

        total += 1

    return {
        f"pass@{k}": pass_at_k[k] / total if total > 0 else 0.0
        for k in k_values
    }


def analyze_reasoning_patterns(predictions: list[dict]) -> dict:
    """Analyze emergent reasoning patterns in model outputs."""
    patterns = {
        "uses_step_numbers": 0,       # "Step 1:", "1.", etc.
        "uses_equations": 0,           # contains = sign with numbers
        "uses_therefore": 0,           # "therefore", "thus", "hence", "so"
        "uses_hashtag_delimiter": 0,   # ####
        "uses_boxed": 0,              # \boxed
        "uses_think_tags": 0,         # <think> or similar
        "shows_intermediate_calc": 0,  # multiple numbers in sequence
        "avg_response_length": 0,
        "avg_num_lines": 0,
    }

    total = len(predictions)
    if total == 0:
        return patterns

    total_length = 0
    total_lines = 0

    for pred in predictions:
        text = pred.get("generated_text", "")
        total_length += len(text)
        lines = [l for l in text.split("\n") if l.strip()]
        total_lines += len(lines)

        if re.search(r"(?:step\s+\d|^\d+[\.\)])", text, re.IGNORECASE | re.MULTILINE):
            patterns["uses_step_numbers"] += 1
        if re.search(r"\d+\s*[+\-*/×÷]\s*\d+\s*=", text):
            patterns["uses_equations"] += 1
        if re.search(r"\b(?:therefore|thus|hence|so,)\b", text, re.IGNORECASE):
            patterns["uses_therefore"] += 1
        if "####" in text:
            patterns["uses_hashtag_delimiter"] += 1
        if "\\boxed" in text:
            patterns["uses_boxed"] += 1
        if "<think>" in text.lower() or "<reasoning>" in text.lower():
            patterns["uses_think_tags"] += 1
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if len(numbers) >= 3:
            patterns["shows_intermediate_calc"] += 1

    patterns["avg_response_length"] = total_length / total
    patterns["avg_num_lines"] = total_lines / total

    # Convert counts to fractions
    for key in patterns:
        if key.startswith("uses_") or key.startswith("shows_"):
            patterns[key] = patterns[key] / total

    return patterns
