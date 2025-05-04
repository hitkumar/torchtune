import gc
import re
import time
from typing import Any, Dict, List, Tuple, Union

import deepspeed
import numpy as np
import torch
from datasets import load_dataset
from deepspeed import DeepSpeedEngine
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-3B"
MODEL_CHAT_NAME = MODEL_NAME + "-Instruct"

# Dataset configuration
DATASET_NAME = "Jiayi-Pan/Countdown-Tasks-3to4"

from prompt_utils import DEFAULT_PROMPT_TEMPLATE, DEFAULT_SYSTEM_MESSAGE
from utils import (
    create_training_episodes,
    GENERATIONS_PER_SAMPLE,
    prepare_model_inputs,
    TEMPERATURE,
)

# Sampling params
MAX_RESPONSE_TOKENS = 1024
TOP_P = 1.0  # disabled nuclear sampling
TOP_K = -1  # no top_k


def preprocess_countdown_example(example: Dict[str, Any], tokenizer: AutoTokenizer):
    numbers: List[int] = example["nums"]
    target: int = example["target"]
    prompt = DEFAULT_PROMPT_TEMPLATE.format(numbers=numbers, target=target)

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


def main():

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_CHAT_NAME)
    EOS_TOKEN_ID = AutoTokenizer.from_pretrained(MODEL_NAME).eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(
        preprocess_countdown_example, num_proc=8, fn_kwargs={"tokenizer": tokenizer}
    )
    train_test_split = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    print(
        f"Train data size is {len(train_dataset)}, test data size is {len(test_dataset)}"
    )

    # Initialize models
    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    # reference_model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.bfloat16,
    #     device_map=0,
    # )

    policy_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
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

    # TODO: Add ability to resume from a checkpoint
    num_samples = 2
    indices = np.random.choice(len(train_dataset), num_samples, replace=False)
    samples = train_dataset.select(indices)
    print(f"Selected {num_samples} samples from the dataset")

    outputs = inference_engine.generate(
        prompt_token_ids=samples["input_ids"],
        sampling_params=SamplingParams(
            n=GENERATIONS_PER_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            detokenize=False,
            stop_token_ids=[EOS_TOKEN_ID],
        ),
    )
    all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
    all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]
    inference_engine.sleep(1)

    print(
        f"Generated {len(all_generations)} generations, one example is:{all_generations[0]}"
    )
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    episodes, episodes_stats = create_training_episodes(
        tokenizer=tokenizer,
        samples=samples,
        all_generations=all_generations,
        all_finish_reasons=all_finish_reasons,
    )
    print(f"Created {len(episodes)} episodes, one example is:{episodes}")
    model_inputs = prepare_model_inputs(training_episodes=episodes, device="cuda")
    print(f"Created {len(model_inputs)} model inputs, one example is:{model_inputs}")


if __name__ == "__main__":
    main()
