#!/usr/bin/env python3
"""Download GSM8K and MATH datasets from HuggingFace."""

import sys
sys.path.insert(0, ".")

from datasets import load_dataset


def main():
    print("=" * 60)
    print("Downloading GSM8K dataset...")
    print("=" * 60)
    ds_gsm = load_dataset("openai/gsm8k", "main")
    print(f"  Train: {len(ds_gsm['train'])} samples")
    print(f"  Test:  {len(ds_gsm['test'])} samples")

    print()
    print("=" * 60)
    print("Downloading MATH dataset (all subjects)...")
    print("=" * 60)
    subjects = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    total_test = 0
    for subj in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subj)
        n_train = len(ds["train"])
        n_test = len(ds["test"])
        total_test += n_test
        print(f"  {subj}: train={n_train}, test={n_test}")
    print(f"  Total test samples: {total_test}")

    print()
    print("Datasets cached successfully.")
    print()

    # Show sample
    print("─" * 60)
    print("GSM8K sample:")
    print(f"  Q: {ds_gsm['train'][0]['question'][:200]}...")
    print(f"  A: {ds_gsm['train'][0]['answer'][:200]}...")
    print()
    print("MATH sample:")
    print(f"  Q: {ds_math['train'][0]['problem'][:200]}...")
    print(f"  A: {ds_math['train'][0]['solution'][:200]}...")


if __name__ == "__main__":
    main()
