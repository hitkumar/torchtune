import argparse
import gc
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    compute_pg_loss,
    create_prompt,
    create_training_episodes,
    dump_episodes,
    dump_training_metrics,
    evaluate_on_test_set,
    get_latest_ckpt,
    load_model_into_vllm,
    prepare_model_inputs,
)
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-3B"
MODEL_CHAT_NAME = MODEL_NAME + "-Instruct"

# Dataset configuration
DATASET_NAME = "Jiayi-Pan/Countdown-Tasks-3to4"

# Define all the constants
MAX_RESPONSE_TOKENS = 1024
TOP_P = 1.0  # disabled nuclear sampling
TOP_K = -1  # no top_k
NUM_ITERATIONS = 1000
EPISODES_PER_ITERATION = 64
PER_DEVICE_BATCH_SIZE = 4
LEARNING_RATE = 1e-6
CONTINUE_TRAINING = False
GENERATIONS_PER_SAMPLE = 4
TEMPERATURE = 1.0
KL_COEFFICIENT = 0.001


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train R1 model with PPO")
    parser.add_argument(
        "--kl_coefficient",
        type=float,
        default=KL_COEFFICIENT,
        help="KL coefficient for GRPO",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-3B", help="Model name/path"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate for training",
    )
    args = parser.parse_args()

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

    # Initialize models
    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=0,
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=0,
    )
    for param in reference_model.parameters():
        param.requires_grad = False

    policy_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
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

    begin_iter = 0
    if CONTINUE_TRAINING:
        ckpt_dir, ckpt_iter = get_latest_ckpt(EXP_DIR / "checkpoints")
        if ckpt_dir is not None and ckpt_iter is not None:
            print(f"Loading checkpoint from {ckpt_dir}")
            begin_iter = ckpt_iter + 1
            try:
                policy_model = AutoModelForCausalLM.from_pretrained(
                    ckpt_dir,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map=0,
                )

                # Initialize optimizer
                optimizer = torch.optim.AdamW(
                    policy_model.parameters(),
                    lr=args.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.0,
                )

                # Load optimizer state if it exists
                optimizer_state_path = Path(ckpt_dir) / "optimizer.pt"
                if optimizer_state_path.exists():
                    optimizer.load_state_dict(torch.load(str(optimizer_state_path)))
                    print(f"Loaded optimizer state from {optimizer_state_path}")
                load_model_into_vllm(policy_model, inference_engine)
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

    for iteration in trange(begin_iter, NUM_ITERATIONS):
        metrics: Dict[str, List[float]] = {}

        if iteration % 100 == 0:
            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                eval_sampling_params=SamplingParams(
                    n=1,
                    temperature=args.temperature,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    detokenize=False,
                    stop_token_ids=[EOS_TOKEN_ID],
                    max_tokens=MAX_RESPONSE_TOKENS,
                ),
            )
            dump_episodes(
                episodes=eval_episodes,
                episode_stats=eval_stats,
                exp_dir=EXP_DIR,
                iteration=iteration,
                tokenizer=tokenizer,
            )

        num_samples = EPISODES_PER_ITERATION // GENERATIONS_PER_SAMPLE
        indices = np.random.choice(len(train_dataset), num_samples, replace=False)
        samples = train_dataset.select(indices)

        # Generate generations from the current policy model
        outputs = inference_engine.generate(
            prompt_token_ids=samples["input_ids"],
            sampling_params=SamplingParams(
                n=GENERATIONS_PER_SAMPLE,
                temperature=args.temperature,
                top_p=TOP_P,
                top_k=TOP_K,
                detokenize=False,
                stop_token_ids=[EOS_TOKEN_ID],
                max_tokens=MAX_RESPONSE_TOKENS,
            ),
        )
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]
        inference_engine.sleep(1)

        # print(
        #     f"Generated {len(all_generations)} generations, one example is:{all_generations[0]}"
        # )
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)  # pause brielfy to avoid memory to clear.

        episodes, episodes_stats = create_training_episodes(
            tokenizer=tokenizer,
            samples=samples,
            all_generations=all_generations,
            all_finish_reasons=all_finish_reasons,
            generations_per_sample=GENERATIONS_PER_SAMPLE,
        )
        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(
                v
            )  # extend list of metrics as these will be aggregated later.

        # print(f"Number of episodes generated: {len(episodes['all_query_token_ids'])}")
        model_inputs = prepare_model_inputs(training_episodes=episodes, device="cuda")
        # print(
        #     f"""input_ids shape: {model_inputs['input_ids'].shape},
        #     labels shape: {model_inputs['labels'].shape},
        #     attention mask shape: {model_inputs['attention_mask'].shape},
        #     advantages shape: {model_inputs['advantages'].shape}"""
        # )
        # print(
        #     f"sum of advantages is {model_inputs['advantages'].sum().item()}, number of non zero advantages is {model_inputs['advantages'].count_nonzero().item()}"
        # )

        # Train the policy model

        policy_model.train()
        reference_model.eval()
        # total number of valid tokens in the batch.
        total_response_len = (model_inputs["labels"] != -100).sum().item()
        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_policy_loss = 0.0
        accumulated_kl_penalty = 0.0
        accumulated_entropy = 0.0

        for i in trange(
            0, EPISODES_PER_ITERATION, PER_DEVICE_BATCH_SIZE, desc="Grad_acc"
        ):
            batch = {
                k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, loss_metrics = compute_pg_loss(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    input_batch=batch,
                    total_response_len=total_response_len,
                    temperature=args.temperature,
                    kl_coefficient=args.kl_coefficient,
                )
                # print(f"loss is {loss}, loss metrics are {loss_metrics}")

            # we are not scaling the loss here by grad_acc steps as we are dividing the loss by total_response_len in `compute_pg_loss`
            loss.backward()
            accumulated_loss += loss.item()
            accumulated_policy_loss += loss_metrics["policy_loss"]
            accumulated_kl_penalty += loss_metrics["kl_distance"]
            accumulated_entropy += loss_metrics["entropy_policy"]

            # print(f"Accumulated loss at step {i} is {accumulated_loss}")
            # Free memory
            del loss, loss_metrics, batch
            gc.collect()
            torch.cuda.empty_cache()

        # optimization step
        # clip gradients
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)
        train_metrics = {
            "loss": accumulated_loss,
            "policy_loss": accumulated_policy_loss,
            "kl_penalty": accumulated_kl_penalty,
            "entropy": accumulated_entropy,
        }
        for k, v in metrics.items():
            train_metrics[k] = np.mean(v)

        train_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]

        dump_training_metrics(train_metrics, iteration, metrics_file)

        if iteration % 100 == 0 or iteration == NUM_ITERATIONS - 1:
            ckpt_save_dir = EXP_DIR / "checkpoints" / f"ckpt_{iteration}"
            ckpt_save_dir.mkdir(parents=True, exist_ok=True)
            policy_model.save_pretrained(str(ckpt_save_dir))
            # Save optimizer state
            torch.save(optimizer.state_dict(), str(ckpt_save_dir / "optimizer.pt"))


if __name__ == "__main__":
    main()
