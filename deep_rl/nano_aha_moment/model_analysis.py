import os
from pathlib import Path

import torch

scratch = Path.home() / "scratch"
os.environ["HF_HOME"] = str(scratch / "hf_home")

from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_OR_NAME = "McGill-NLP/nano-aha-moment-3b"
CHAT_MODEL_NAME = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(
    CHAT_MODEL_NAME, cache_dir=scratch / "hf_cache"
)
print(f"tokenizer len is {len(tokenizer)}")
model = AutoModelForCausalLM.from_pretrained(
    CHAT_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    cache_dir=scratch / "hf_home",
)

import torch
from vllm import LLM, SamplingParams

inference_engine = LLM(
    model=CHAT_MODEL_NAME,
    gpu_memory_utilization=0.5,
    dtype=torch.bfloat16,
    swap_space=2,
    enable_prefix_caching=True,
    max_model_len=2048,
    max_seq_len_to_capture=2048,
)
from prompt_utils import DEFAULT_PROMPT_TEMPLATE, DEFAULT_SYSTEM_MESSAGE

print(inference_engine)


def preprocess_countdown_example(example):

    numbers = example["nums"]
    target = example["target"]
    prompt = DEFAULT_PROMPT_TEMPLATE.format(numbers=numbers, target=target)
    print(f"prompt is {prompt}")

    chat_messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Let me think step by step\n<think>"},
    ]

    input_ids = tokenizer.apply_chat_template(
        chat_messages, tokenize=True, continue_final_message=True
    )
    prompt = tokenizer.decode(
        input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return {
        "input_ids": input_ids,
        "prompt": prompt,
    }


sample = {"nums": [7, 19, 4], "target": 69}
sample.update(preprocess_countdown_example(sample))
print(f"sample is {sample}")

generation = inference_engine.generate(
    prompt_token_ids=sample["input_ids"],
    sampling_params=SamplingParams(temperature=0.5, max_tokens=2048, top_p=1.0, n=1),
)
response = tokenizer.decode(generation[0].outputs[0].token_ids)

print(f"response is {response}")

# Response from Qwen 2.5 instruct model.
# response is  We need to create an equation using the numbers 7, 19, and 4 exactly once to equal 69. Let's consider how we can manipulate these numbers using basic arithmetic operations. We can start by trying to use multiplication and addition since these operations can help us reach the target number 69 more efficiently. If we multiply 19 by 3, we get 57, which is close to 69. Then, we need an additional 12 to reach 69. Using the remaining numbers 7 and 4, we can achieve this by adding 7 and 4 to 57. However, we need to check if this combination uses each number exactly once and if it can be formulated as a valid equation. </think>
# <answer>(19 * 3) + 7 + 4</answer><|im_end|>


# Response from nano-aha-moment-3b model.
# response is  Start with the largest number, 19. Add 7 to it: 19 + 7 = 26. Now, add 4 to the result: 26 + 4 = 30. However, this doesn't work because 30 is less than 69. Let's try a different order. Start with the smallest number, 4. Multiply it by 19: 4 * 19 = 76. Finally, subtract 7: 76 - 7 = 69. This works!</think>
# <answer>(4 * 19) - 7</answer><|endoftext|>
