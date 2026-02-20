#!/usr/bin/env python3
"""Generate example reasoning traces from trained models."""

import sys
sys.path.insert(0, ".")

import argparse
import json
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.grpo_reasoning.data.dataset import SYSTEM_PROMPT
from src.grpo_reasoning.utils.config import load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


EXAMPLE_PROBLEMS = [
    {
        "question": "A store sells notebooks for $3 each. If you buy 5 or more, you get a 20% discount. How much would 7 notebooks cost?",
        "answer": "16.80",
        "source": "custom",
    },
    {
        "question": "A train travels at 60 mph for 2 hours, then at 80 mph for 1.5 hours. What is the total distance traveled?",
        "answer": "240",
        "source": "custom",
    },
    {
        "question": "If 3x + 7 = 22, what is the value of x?",
        "answer": "5",
        "source": "custom",
    },
    {
        "question": "A rectangular garden has a perimeter of 56 meters. If the length is 4 meters more than the width, what is the area of the garden?",
        "answer": "192",
        "source": "custom",
    },
    {
        "question": "In a class of 30 students, 18 play basketball, 15 play soccer, and 5 play neither. How many students play both sports?",
        "answer": "8",
        "source": "custom",
    },
]


def generate_trace(model, tokenizer, question: str, temperature: float = 0.0,
                   max_new_tokens: int = 1024) -> str:
    """Generate a reasoning trace for a question."""
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to LoRA adapter")
    parser.add_argument("--label", type=str, default="model",
                        help="Model label for output")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["grpo"]["seed"])
    base_model_name = config["model"]["name"]

    logger.info(f"Loading model: {base_model_name}")
    logger.info(f"Adapter: {args.adapter}")

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

    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    traces = []
    for i, problem in enumerate(EXAMPLE_PROBLEMS, 1):
        logger.info(f"\n{'─'*60}")
        logger.info(f"Problem {i}: {problem['question'][:80]}...")
        trace = generate_trace(model, tokenizer, problem["question"], args.temperature)
        logger.info(f"Response:\n{trace}")
        logger.info(f"Expected answer: {problem['answer']}")
        traces.append({
            "question": problem["question"],
            "expected_answer": problem["answer"],
            "model_response": trace,
        })

    output_path = Path(config["paths"]["results_dir"]) / f"{args.label}_traces.json"
    with open(output_path, "w") as f:
        json.dump(traces, f, indent=2)
    logger.info(f"\nTraces saved to {output_path}")


if __name__ == "__main__":
    main()
