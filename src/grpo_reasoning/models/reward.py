"""Reward functions for GRPO training."""

import re
from src.grpo_reasoning.data.dataset import extract_model_answer, normalize_answer


def _to_str(x) -> str:
    """Coerce completion/answer to a plain string."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        # Could be a list of message dicts or list of strings
        parts = []
        for item in x:
            if isinstance(item, dict):
                parts.append(item.get("content", ""))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(x)


def correctness_reward(completion, answer, **kwargs) -> float:
    """Binary reward: 1.0 if the model's answer matches the ground truth, else 0.0."""
    completion = _to_str(completion)
    answer = _to_str(answer)
    model_answer = extract_model_answer(completion)
    model_answer = normalize_answer(model_answer)
    expected = normalize_answer(answer)

    if model_answer is None or expected is None:
        return 0.0

    return 1.0 if model_answer == expected else 0.0


def format_reward(completion, **kwargs) -> float:
    """Reward for structured reasoning format.

    Gives partial credit for:
    - Showing step-by-step work (multiple lines with numbers)
    - Using #### to denote final answer
    - Having reasoning before the answer
    """
    completion = _to_str(completion)
    score = 0.0

    # Has multiple lines (shows work)
    lines = [l.strip() for l in completion.strip().split("\n") if l.strip()]
    if len(lines) >= 3:
        score += 0.3

    # Contains numerical steps
    num_lines_with_numbers = sum(
        1 for l in lines if re.search(r"\d+", l)
    )
    if num_lines_with_numbers >= 2:
        score += 0.2

    # Uses #### delimiter for final answer
    if "####" in completion:
        score += 0.3

    # Has reasoning BEFORE the final answer
    if "####" in completion:
        parts = completion.split("####")
        if len(parts[0].strip()) > 50:
            score += 0.2

    return min(score, 1.0)


def combined_reward(
    completion: str,
    answer: str,
    correctness_weight: float = 1.0,
    format_weight: float = 0.1,
    **kwargs,
) -> float:
    """Combined reward: correctness + format bonus."""
    c_reward = correctness_reward(completion, answer)
    f_reward = format_reward(completion)
    return correctness_weight * c_reward + format_weight * f_reward


def build_reward_fn(config: dict):
    """Build the reward function from config.

    Returns a function compatible with TRL GRPOTrainer:
    reward_fn(completions, **kwargs) -> list[float]
    """
    c_weight = config["reward"]["correctness_weight"]
    f_weight = config["reward"]["format_weight"]

    def reward_fn(completions: list[str], answer: list[str], **kwargs) -> list[float]:
        rewards = []
        for comp, ans in zip(completions, answer):
            r = combined_reward(
                comp, ans,
                correctness_weight=c_weight,
                format_weight=f_weight,
            )
            rewards.append(r)
        return rewards

    return reward_fn
