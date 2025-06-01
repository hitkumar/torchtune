import pytest

from ..reward_functions import (
    compute_reward,
    EOS_TOKEN,
    equation_reward_func,
    format_reward_func,
)


class TestFormatRewardFunc:
    def test_valid_format_full_reward(self):
        """Test a valid format with proper think/answer tags and valid answer content."""
        completion = "Let me solve this problem\nStep 1: Add the numbers\nStep 2: Multiply</think>\n<answer>2 + 3 * 4</answer>"
        reward = format_reward_func(completion)
        assert reward == 1.0

    def test_valid_format_partial_reward(self):
        """Test a valid format but with non-mathematical content in answer."""
        completion = "Let me solve this</think>\n<answer>This is the answer</answer>"
        reward = format_reward_func(completion)
        assert reward == 0.5

    def test_missing_think_tag(self):
        """Test with missing think tag."""
        completion = "Let me solve this\n<answer>2 + 3 * 4</answer>"
        reward = format_reward_func(completion)
        assert reward == 0.0

    def test_missing_answer_tag(self):
        """Test with missing answer tag."""
        completion = "Let me solve this</think>\nThe answer is 2 + 3 * 4"
        reward = format_reward_func(completion)
        assert reward == 0.0

    def test_with_eos_token(self):
        """Test with EOS token at the end."""
        completion = "Let me solve this</think>\n<answer>2 + 3 * 4</answer>" + EOS_TOKEN
        reward = format_reward_func(completion)
        assert reward == 1.0

    def test_malformed_tags(self):
        """Test with malformed tags."""
        completion = "Let me solve this<think>\n<answer>2 + 3 * 4</answer>"
        reward = format_reward_func(completion)
        assert reward == 0.0

    def test_empty_answer(self):
        """Test with empty answer."""
        completion = "Let me solve this</think>\n<answer></answer>"
        reward = format_reward_func(completion)
        assert reward == 0.5  # Empty string matches the allowed pattern

    def test_complex_math_expression(self):
        """Test with a complex mathematical expression."""
        completion = "Let me solve this</think>\n<answer>2 + 3 * (4 - 1) / 2.5</answer>"
        reward = format_reward_func(completion)
        assert reward == 1.0


class TestEquationRewardFunc:
    def test_correct_equation(self):
        """Test with a correct equation using all required numbers."""
        completion = (
            "I need to use 2, 3, and 4 to get 10</think>\n<answer>2 + 4 * 2</answer>"
        )
        nums = [2, 4, 2]
        target = 10
        reward = equation_reward_func(completion, nums, target)
        assert reward == 1.0

    def test_incorrect_result(self):
        """Test with an equation that doesn't evaluate to the target."""
        completion = (
            "I need to use 2, 3, and 4 to get 10</think>\n<answer>2 + 3 + 4</answer>"
        )
        nums = [2, 3, 4]
        target = 10
        reward = equation_reward_func(completion, nums, target)
        assert reward == 0.0

    def test_missing_numbers(self):
        """Test with an equation missing some required numbers."""
        completion = (
            "I need to use 2, 3, and 4 to get 10</think>\n<answer>2 + 8</answer>"
        )
        nums = [2, 3, 4]
        target = 10
        reward = equation_reward_func(completion, nums, target)
        assert reward == 0.0

    def test_extra_numbers(self):
        """Test with an equation using numbers not in the list."""
        completion = (
            "I need to use 2, 3, and 4 to get 10</think>\n<answer>2 + 3 + 5</answer>"
        )
        nums = [2, 3, 4]
        target = 10
        reward = equation_reward_func(completion, nums, target)
        assert reward == 0.0

    def test_invalid_equation(self):
        """Test with an invalid equation containing non-mathematical symbols."""
        completion = "I need to use 2, 3, and 4 to get 10</think>\n<answer>2 + 3 + 4 = 9</answer>"
        nums = [2, 3, 4]
        target = 9
        reward = equation_reward_func(completion, nums, target)
        assert reward == 0.0

    def test_close_to_target(self):
        """Test with an equation that's very close to the target (floating point comparison)."""
        completion = (
            "I need to use numbers to get close to 3</think>\n<answer>10 / 3</answer>"
        )
        nums = [10, 3]
        target = 3
        reward = equation_reward_func(completion, nums, target)
        assert reward == 0.0

    def test_missing_answer_tag(self):
        """Test with missing answer tag."""
        completion = "Let me solve this</think>\nThe answer is 2 + 3 * 4"
        nums = [2, 3, 4]
        target = 14
        reward = equation_reward_func(completion, nums, target)
        assert reward == 0.0

    def test_duplicate_numbers(self):
        """Test with duplicate numbers in the input list."""
        completion = "I need to use the numbers</think>\n<answer>2 + 2 + 3</answer>"
        nums = [2, 2, 3]
        target = 7
        reward = equation_reward_func(completion, nums, target)
        assert reward == 1.0


class TestComputeReward:
    def test_both_rewards_full(self):
        """Test when both format and equation rewards are full."""
        completion = (
            "I need to use 2, 3, and 4 to get 14</think>\n<answer>2 + 3 * 4</answer>"
        )
        sample = {"nums": [2, 3, 4], "target": 14}
        reward, metrics = compute_reward(completion, sample)
        assert metrics["format_reward"] == 1.0
        assert metrics["equation_reward"] == 1.0
        assert reward == 2.0

    def test_format_full_equation_zero(self):
        """Test when format reward is full but equation reward is zero."""
        completion = (
            "I need to use 2, 3, and 4 to get 10</think>\n<answer>2 + 3 * 4</answer>"
        )
        sample = {"nums": [2, 3, 4], "target": 10}
        reward, metrics = compute_reward(completion, sample)
        assert metrics["format_reward"] == 1.0
        assert metrics["equation_reward"] == 0.0
        assert reward == 1.0

    def test_format_partial_equation_zero(self):
        """Test when format reward is partial and equation reward is zero."""
        completion = "I need to solve this</think>\n<answer>This is not a valid equation</answer>"
        sample = {"nums": [2, 3, 4], "target": 14}
        reward, metrics = compute_reward(completion, sample)
        assert metrics["format_reward"] == 0.5
        assert metrics["equation_reward"] == 0.0
        assert reward == 0.5

    def test_both_rewards_zero(self):
        """Test when both rewards are zero."""
        completion = "This doesn't have the right format or equation"
        sample = {"nums": [2, 3, 4], "target": 14}
        reward, metrics = compute_reward(completion, sample)
        assert metrics["format_reward"] == 0.0
        assert metrics["equation_reward"] == 0.0
        assert reward == 0.0

    def test_with_eos_token(self):
        """Test with EOS token at the end."""
        completion = (
            "I need to use 2, 3, and 4 to get 14</think>\n<answer>2 + 3 * 4</answer>"
            + EOS_TOKEN
        )
        sample = {"nums": [2, 3, 4], "target": 14}
        reward, metrics = compute_reward(completion, sample)
        assert metrics["format_reward"] == 1.0
        assert metrics["equation_reward"] == 1.0
        assert reward == 2.0
