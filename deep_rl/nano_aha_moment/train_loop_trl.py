# Run using the following command:
# accelerate launch --num_processes 8 --config_file deepspeed_zero3.yaml train_loop_trl.py --config training_args.yaml > /tmp/logs.txt 2>&1
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset
from reward_functions import equation_reward_func, format_reward_func
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import get_peft_config, GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from utils import DEFAULT_PROMPT_TEMPLATE, DEFAULT_SYSTEM_MESSAGE


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
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    print(f"Training args: {training_args}")

    # Set HF cache directory
    os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), "scratch", "hf_home")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    dataset = load_dataset(
        script_args.dataset_id_or_path, split=script_args.dataset_splits
    )
    dataset = dataset.map(create_prompt, num_proc=8, fn_kwargs={"tokenizer": tokenizer})
    train_test_split = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    print(
        f"Train data size is {len(train_dataset)}, test data size is {len(test_dataset)}"
    )
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
