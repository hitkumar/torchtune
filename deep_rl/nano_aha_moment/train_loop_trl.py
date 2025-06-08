# Run using the following command:
# accelerate launch --num_processes 8 --config_file deepspeed_zero3.yaml train_loop_trl.py --config training_args.yaml > /tmp/logs.txt 2>&1
import argparse
import gc
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from reward_functions import equation_reward_func, format_reward_func
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import get_peft_config, GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from utils import DEFAULT_PROMPT_TEMPLATE, DEFAULT_SYSTEM_MESSAGE
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-3B"
MODEL_CHAT_NAME = MODEL_NAME + "-Instruct"

# Dataset configuration
DATASET_NAME = "Jiayi-Pan/Countdown-Tasks-3to4"

# Define all the constants
MAX_RESPONSE_TOKENS = 1024
TOP_P = 1.0  # disabled nuclear sampling
TOP_K = -1  # no top_k
NUM_ITERATIONS = 3000
EPISODES_PER_ITERATION = 64
PER_DEVICE_BATCH_SIZE = 4
LEARNING_RATE = 1e-6 * 1.5
CONTINUE_TRAINING = True
GENERATIONS_PER_SAMPLE = 4
TEMPERATURE = 1.0
KL_COEFFICIENT = 0.001


def create_prompt(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
):
    """
    Create a prompt for a given example.
    """
    numbers: List[int] = example["nums"]
    target: int = example["target"]
    prompt = DEFAULT_PROMPT_TEMPLATE.format(numbers=numbers, target=target)

    chat_messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Let me think step by step\n<think>"},
    ]

    input_ids = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, continue_final_message=True
    )
    return {
        "prompt": input_ids,
        "target": target,
    }


def group_format_reward_func(completions, target, **kwargs):
    """
    Group format reward function.
    """
    format_rewards = [format_reward_func(completion) for completion in completions]
    for i in range(len(completions)):
        if (
            random.random() < 0.10
        ):  # 10% chance to write fully successful samples into a file
            os.makedirs("completion_samples", exist_ok=True)
            log_file = os.path.join("completion_samples", "completion_samples.txt")
            with open(log_file, "a") as f:
                f.write(f"\n\n==============\n")
                f.write(completions[i])
    return format_rewards


def group_equation_reward_func(completions, target, nums, **kwargs):
    """
    Group equation reward function.
    """
    equation_rewards = [
        equation_reward_func(completion, nums, target) for completion in completions
    ]
    for i in range(len(equation_rewards)):
        if equation_rewards[i] == 1.0:
            if (
                random.random() < 0.10
            ):  # 10% chance to write fully successful samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join(
                    "completion_samples", "success_completion_samples.txt"
                )
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completions[i])
    return equation_rewards


@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


def get_checkpoint(training_args: GRPOConfig):
    if os.path.isdir(training_args.output_dir):
        return get_last_checkpoint(training_args.output_dir)
    return None


def main():

    # Set the environment variables for HuggingFace
    # This is done to ensure that the cache directory for HuggingFace is set to a specific location,
    # preventing the storage from being overwhelmed with model files and other data.
    SCRATCH = Path.home() / "scratch"
    os.environ["HF_HOME"] = str(SCRATCH / "hf_home")
    RUN_NAME = "r1-zero-v2"
    EXP_DIR = SCRATCH / RUN_NAME
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = EXP_DIR / "train_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "metrics.jsonl"

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_CHAT_NAME)
    EOS_TOKEN_ID = AutoTokenizer.from_pretrained(MODEL_NAME).eos_token_id

    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(create_prompt, num_proc=8, fn_kwargs={"tokenizer": tokenizer})
    train_test_split = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    print(
        f"Train data size is {len(train_dataset)}, test data size is {len(test_dataset)}"
    )

    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    print(f"Training args: {training_args}")
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[group_equation_reward_func, group_format_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
    )

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        print(f"Resuming training from checkpoint {last_checkpoint}")

    print(f"Starting training with {training_args.num_train_epochs} iterations")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("*** Train completed***")

    print("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Tokenizer saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
