from typing import Any, Callable, Dict, List, Tuple, Union

from transformers import AutoTokenizer

DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process in the mind "
    "and then provides the user with the answer."
)

DEFAULT_PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. And return the final equation and answer in "
    "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
)


def create_prompt(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    prompt_message: str = DEFAULT_PROMPT_TEMPLATE,
):
    """
    Create a prompt for a given example.
    """
    numbers: List[int] = example["nums"]
    target: int = example["target"]
    user_prompt = prompt_message.format(numbers=numbers, target=target)
    chat_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Let me think step by step\n<think>"},
    ]
    # Check if tokenize should be true here.
    return tokenizer.apply_chat_template(
        chat_messages, tokenize=False, continue_final_message=True
    )
