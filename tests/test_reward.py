"""Tests for reward functions and answer extraction."""

import pytest
from src.grpo_reasoning.data.dataset import (
    extract_gsm8k_answer,
    extract_math_answer,
    extract_model_answer,
    normalize_answer,
)
from src.grpo_reasoning.models.reward import (
    combined_reward,
    correctness_reward,
    format_reward,
)


class TestAnswerExtraction:
    def test_gsm8k_answer_basic(self):
        assert extract_gsm8k_answer("The answer is #### 42") == "42"

    def test_gsm8k_answer_with_comma(self):
        assert extract_gsm8k_answer("Total: #### 1,234") == "1234"

    def test_gsm8k_answer_negative(self):
        assert extract_gsm8k_answer("Result: #### -5") == "-5"

    def test_gsm8k_answer_decimal(self):
        assert extract_gsm8k_answer("Price is #### 16.80") == "16.80"

    def test_gsm8k_answer_none(self):
        assert extract_gsm8k_answer("no answer here") is None

    def test_math_answer_basic(self):
        assert extract_math_answer("So \\boxed{42}") == "42"

    def test_math_answer_fraction(self):
        assert extract_math_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_math_answer_none(self):
        assert extract_math_answer("no boxed answer") is None

    def test_model_answer_hashtag(self):
        assert extract_model_answer("Step 1: 5*3=15\n#### 15") == "15"

    def test_model_answer_boxed(self):
        assert extract_model_answer("Therefore \\boxed{7}") == "7"

    def test_model_answer_text(self):
        assert extract_model_answer("The answer is 42") == "42"

    def test_model_answer_final_answer_is(self):
        assert extract_model_answer("So the final answer is 99") == "99"

    def test_model_answer_last_number(self):
        assert extract_model_answer("Computing: 3+4=7, then 7*2=14") == "14"

    def test_model_answer_empty(self):
        assert extract_model_answer("no numbers here at all") is None


class TestNormalization:
    def test_normalize_int(self):
        assert normalize_answer("42") == "42"

    def test_normalize_float_whole(self):
        assert normalize_answer("42.0") == "42"

    def test_normalize_float(self):
        assert normalize_answer("16.80") == "16.8"

    def test_normalize_comma(self):
        assert normalize_answer("1,234") == "1234"

    def test_normalize_none(self):
        assert normalize_answer(None) is None

    def test_normalize_text(self):
        assert normalize_answer("x+1") == "x+1"


class TestCorrectnessReward:
    def test_correct(self):
        assert correctness_reward("#### 42", "42") == 1.0

    def test_incorrect(self):
        assert correctness_reward("#### 43", "42") == 0.0

    def test_no_answer(self):
        assert correctness_reward("I don't know", "42") == 0.0

    def test_correct_with_comma(self):
        assert correctness_reward("#### 1,234", "1234") == 1.0

    def test_correct_float_int(self):
        assert correctness_reward("#### 42.0", "42") == 1.0


class TestFormatReward:
    def test_no_reasoning(self):
        assert format_reward("42") == 0.0

    def test_multiline_with_hashtag(self):
        text = "Step 1: Calculate base.\nBase = 3 * 7 = 21\nAdd tax: 21 + 0.8 = 21.8\n#### 21.8"
        score = format_reward(text)
        assert score >= 0.5

    def test_short_response(self):
        score = format_reward("The answer is 5")
        assert score < 0.5

    def test_full_reasoning(self):
        text = (
            "Let me solve this step by step.\n"
            "Step 1: Find the base price = 3 * 7 = 21\n"
            "Step 2: Apply the discount = 21 * 0.8 = 16.8\n"
            "Step 3: The total cost is $16.80\n"
            "#### 16.80"
        )
        score = format_reward(text)
        assert score >= 0.8


class TestCombinedReward:
    def test_correct_with_format(self):
        text = "Step 1: 5*3=15\nStep 2: 15*2=30\nSo the answer is:\n#### 30"
        reward = combined_reward(text, "30", correctness_weight=1.0, format_weight=0.1)
        assert reward > 1.0  # correctness + format bonus

    def test_wrong_with_format(self):
        text = "Step 1: 5*3=15\nStep 2: 15*2=30\nSo the answer is:\n#### 29"
        reward = combined_reward(text, "30", correctness_weight=1.0, format_weight=0.1)
        assert reward < 1.0  # no correctness, only format
        assert reward > 0.0  # but some format bonus
