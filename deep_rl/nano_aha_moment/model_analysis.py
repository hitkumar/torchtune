import os
from pathlib import Path

import numpy as np

import torch
from datasets import load_dataset

from train_loop import (
    DATASET_NAME,
    MAX_RESPONSE_TOKENS,
    MODEL_CHAT_NAME,
    MODEL_NAME,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from utils import create_prompt, evaluate_on_test_set, load_model_into_vllm
from vllm import LLM, SamplingParams

scratch = Path.home() / "scratch"
os.environ["HF_HOME"] = str(scratch / "hf_home")

from transformers import AutoModelForCausalLM, AutoTokenizer


def eval_on_sample(sample, CHECKPOINT_OR_NAME, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_OR_NAME,
        torch_dtype=torch.bfloat16,
        cache_dir=scratch / "hf_home",
    )
    inference_engine = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.5,
        dtype=torch.bfloat16,
        swap_space=2,
        enable_prefix_caching=True,
        max_model_len=1024,
        max_seq_len_to_capture=1024,
    )
    load_model_into_vllm(model, inference_engine)
    print(inference_engine)

    prompt = create_prompt(sample, tokenizer)
    print(f"prompt is {prompt}")

    generation = inference_engine.generate(
        prompt_token_ids=prompt["input_ids"],
        sampling_params=SamplingParams(
            temperature=1.0, max_tokens=2048, top_p=1.0, n=1
        ),
    )
    response = tokenizer.decode(generation[0].outputs[0].token_ids)
    print(f"response is {response}")


def eval_testset(
    test_dataset,
    model_name,
    tokenizer,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=scratch / "hf_home",
    )
    inference_engine = LLM(
        model=MODEL_NAME,
        skip_tokenizer_init=False,
        gpu_memory_utilization=0.2,
        enable_prefix_caching=True,
        swap_space=1,
        scheduling_policy="fcfs",
        dtype=torch.bfloat16,
        max_model_len=2048,
        enable_sleep_mode=True,
    )
    load_model_into_vllm(model, inference_engine)
    _, eval_stats = evaluate_on_test_set(
        inference_engine=inference_engine,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        eval_sampling_params=SamplingParams(
            n=1,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            detokenize=False,
            stop_token_ids=[tokenizer.eos_token_id],
            max_tokens=MAX_RESPONSE_TOKENS,
        ),
    )
    reward_avg = np.mean(eval_stats["rewards"])
    response_len_avg = np.mean(eval_stats["response_length"])

    return (reward_avg, response_len_avg)


if __name__ == "__main__":
    CHECKPOINT_OR_NAME = "McGill-NLP/nano-aha-moment-3b"
    # CHECKPOINT_OR_NAME = "htkumar/nan_aha_moment_900"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CHAT_NAME, cache_dir=scratch / "hf_cache"
    )
    # sample = {"nums": [7, 19, 4], "target": 69}
    # eval_on_sample(sample, MODEL_CHAT_NAME, tokenizer)

    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(create_prompt, num_proc=8, fn_kwargs={"tokenizer": tokenizer})
    train_test_split = dataset.train_test_split(test_size=500, seed=42)
    test_dataset = train_test_split["test"]
    # (0.947, 163.402) for htkumar/nan_aha_moment_900
    # (1.608, 382.918) for McGill-NLP/nano-aha-moment-3b
    # ((0.146, 384.276) for Qwen-2.5
    print(f"eval stats is {eval_testset(test_dataset, MODEL_NAME, tokenizer)}")


# Response from Qwen 2.5 instruct model.
# response is  We need to create an equation using the numbers 7, 19, and 4 exactly once to equal 69. Let's consider how we can manipulate these numbers using basic arithmetic operations. We can start by trying to use multiplication and addition since these operations can help us reach the target number 69 more efficiently. If we multiply 19 by 3, we get 57, which is close to 69. Then, we need an additional 12 to reach 69. Using the remaining numbers 7 and 4, we can achieve this by adding 7 and 4 to 57. However, we need to check if this combination uses each number exactly once and if it can be formulated as a valid equation. </think>
# <answer>(19 * 3) + 7 + 4</answer><|im_end|>


# Response from nano-aha-moment-3b model.
# response is  Start with the largest number, 19. Add 7 to it: 19 + 7 = 26. Now, add 4 to the result: 26 + 4 = 30. However, this doesn't work because 30 is less than 69. Let's try a different order. Start with the smallest number, 4. Multiply it by 19: 4 * 19 = 76. Finally, subtract 7: 76 - 7 = 69. This works!</think>
# <answer>(4 * 19) - 7</answer><|endoftext|>
