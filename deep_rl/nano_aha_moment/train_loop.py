import gc
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-3B"
MODEL_CHAT_NAME = MODEL_NAME + "-Instruct"

# Dataset configuration
DATASET_NAME = "Jiayi-Pan/Countdown-Tasks-3to4"

from prompt_utils import DEFAULT_PROMPT_TEMPLATE, DEFAULT_SYSTEM_MESSAGE
from utils import (
    compute_pg_loss,
    create_training_episodes,
    dump_episodes,
    evaluate_on_test_set,
    GENERATIONS_PER_SAMPLE,
    load_model_into_vllm,
    prepare_model_inputs,
    TEMPERATURE,
)

# Sampling params
MAX_RESPONSE_TOKENS = 1024
TOP_P = 1.0  # disabled nuclear sampling
TOP_K = -1  # no top_k
NUM_ITERATIONS = 100
EPISODES_PER_ITERATION = 64
PER_DEVICE_BATCH_SIZE = 4
LEARNING_RATE = 1e-6


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
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    for param in reference_model.parameters():
        param.requires_grad = False

    policy_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=LEARNING_RATE,
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
        max_model_len=MAX_RESPONSE_TOKENS,
        enable_sleep_mode=True,
    )

    # TODO: Add ability to resume from a checkpoint
    begin_iter = 0
    for iteration in trange(begin_iter, NUM_ITERATIONS):

        if iteration % 5 == 0:
            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                eos_token=EOS_TOKEN,
                eval_sampling_params=SamplingParams(
                    n=1,
                    temperature=TEMPERATURE,
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
                exp_dir=Path("/tmp"),
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
                temperature=TEMPERATURE,
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
        time.sleep(1)

        episodes, episodes_stats = create_training_episodes(
            tokenizer=tokenizer,
            samples=samples,
            all_generations=all_generations,
            all_finish_reasons=all_finish_reasons,
        )

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
        total_response_len = (model_inputs["labels"] != -100).sum().item()
        optimizer.zero_grad()
        accumulated_loss = 0.0

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
                )
                # print(f"loss is {loss}, loss metrics are {loss_metrics}")

            loss.backward()
            accumulated_loss += loss.item()
            print(f"Accumulated loss at step {i} is {accumulated_loss}")
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


if __name__ == "__main__":
    main()
