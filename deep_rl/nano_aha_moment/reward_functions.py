import re
from typing import Any, Dict, List, Tuple

EOS_TOKEN = "<|endoftext|>"


def format_reward_func(completion: str) -> float:
    """
    Format: <think>...</think>\n<answer>...</answer>
    """
    allowed_pattern = r"^[\d+\-*/().\s]+$"
    try:
        completion = "<think>" + completion

        if completion.endswith(EOS_TOKEN):
            completion = completion[: -len(EOS_TOKEN)]

        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 2:
            return 0.0
        else:
            answer_content = match.group(2).strip()
            if not re.match(allowed_pattern, answer_content):
                return 0.5
            else:
                return 1.0

    except Exception:
        return 0.0


def equation_reward_func(completion: str, nums: List[int], target: int) -> float:
    try:
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0

        equation = match.group(1).strip()
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        if sorted(used_numbers) != sorted(nums):
            return 0.0

        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        result = eval(equation, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-6:
            return 1.0
        else:
            return 0.0

    except Exception:
        return 0.0


def compute_reward(
    completion: str, sample: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    format_reward = format_reward_func(completion)
    equation_reward = equation_reward_func(completion, sample["nums"], sample["target"])
    # todo: make this weighted?
    reward = 1.0 * format_reward + 1.0 * equation_reward
    metrics = {
        "format_reward": format_reward,
        "equation_reward": equation_reward,
    }
    return reward, metrics
