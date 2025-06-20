import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset

from reward_functions import compute_reward
from transformers import AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

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
        chat_messages, tokenize=True, continue_final_message=True
    )
    prompt = tokenizer.decode(
        input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return {
        "input_ids": input_ids,
    }


def compute_pg_loss(
    policy_model: PreTrainedModel,
    reference_model: PreTrainedModel,
    input_batch: Dict[str, torch.tensor],
    total_response_len: int,
    temperature: float,
    kl_coefficient: float,
) -> Tuple[torch.tensor, Dict[str, float]]:
    """
    Compute the policy gradient loss for the policy model by combining PPO loss and KL penalty.
    """
    # inputs are dim [bsz, seq_len]

    # Create labels mask to properly handle padding and non-response tokens
    labels = input_batch["labels"]
    labels_mask = (labels[..., 1:] != -100).float()  # [bsz, seq_len-1]

    # [bsz, seq_len-1]
    with torch.no_grad():
        ref_model_logprobs = compute_token_log_probs(
            reference_model, input_batch, temperature
        )
    policy_model_logprobs = compute_token_log_probs(
        policy_model, input_batch, temperature
    )

    diff = ref_model_logprobs - policy_model_logprobs
    kl_distance = torch.exp(diff) - diff - 1
    # Apply mask to ensure padding doesn't contribute to KL distance
    kl_distance = kl_distance * labels_mask

    policy_loss = (
        -policy_model_logprobs * input_batch["advantages"][..., 1:]
    )  # [bsz, seq_len-1]
    # Apply mask to ensure padding doesn't contribute to policy loss
    policy_loss = policy_loss * labels_mask

    loss = (
        policy_loss + kl_coefficient * kl_distance
    ).sum() / total_response_len  # scalar

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len,
        "kl_distance": kl_distance.sum().item() / total_response_len,
        # entropy should decrease over time as the policy becomes more certain, for reference model it should stay the same over time.
        "entropy_policy": -policy_model_logprobs.sum().item() / total_response_len,
        "entropy_ref": -ref_model_logprobs.sum().item() / total_response_len,
    }
    return loss, metrics


def compute_token_log_probs(
    model: PreTrainedModel,
    inputs: Dict[str, torch.tensor],
    temperature: float,
) -> torch.tensor:
    """
    Compute the log probabilities of the next token given the input sequence.
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model_output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
            use_cache=False,
        )
    logits = model_output.logits.float() / temperature
    shift_logits = logits[..., :-1, :].contiguous()  # [bsz, seq_len-1, vocab_size]
    shift_labels = inputs["labels"][..., 1:].contiguous()  # [bsz, seq_len-1]

    label_mask = (shift_labels != -100).float()
    shift_labels[shift_labels == -100] = 0

    # [bsz, seq_len-1, vocab_size]
    log_probs = torch.log_softmax(shift_logits, dim=-1)  # log(softmax(logits))

    # [bsz, seq_len-1]
    log_probs = torch.gather(
        log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    # [bsz, seq_len-1]
    log_probs = log_probs * label_mask
    return log_probs


def prepare_model_inputs(
    training_episodes: Dict[str, Any], device: torch.device
) -> Dict[str, torch.tensor]:
    query_token_ids = training_episodes["all_query_token_ids"]
    response_token_ids = training_episodes["all_response_token_ids"]
    advantages = training_episodes["all_advantages"]
    # print(len(query_token_ids), len(response_token_ids), len(advantages))

    max_seq_len = max(
        len(q) + len(r) for q, r in zip(query_token_ids, response_token_ids)
    )
    input_ids, attention_mask, labels, advantage_list = [], [], [], []
    pad_token_id = 0
    ignore_index = -100  # check nn.CrossEntropyLoss for more context

    for q, r, a in zip(query_token_ids, response_token_ids, advantages):
        # print(q)
        combined_ids = q + r
        seq_len = len(combined_ids)
        # print(f"seq_len is {len(seq_len)}")
        input_ids.append(combined_ids + [pad_token_id] * (max_seq_len - seq_len))
        attention_mask.append([1] * seq_len + [0] * (max_seq_len - seq_len))
        labels.append(
            [ignore_index] * len(q) + r + [ignore_index] * (max_seq_len - seq_len)
        )
        advantage_list.append([0.0] * len(q) + a + [0.0] * (max_seq_len - seq_len))

    # print(len(input_ids))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "advantages": torch.tensor(advantage_list, dtype=torch.float, device=device),
    }


def create_training_episodes(
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, Any]],
    all_generations: List[List[int]],
    all_finish_reasons: List[str],
    generations_per_sample: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create training episodes from samples and generations.
    """

    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * generations_per_sample
    groups = [
        list(range(i, i + generations_per_sample))
        for i in range(0, len(all_generations), generations_per_sample)
    ]

    all_query_token_ids, all_response_token_ids, all_advantages = [], [], []
    stats = {"response_length": [], "non_stop_rate": [], "rewards": []}

    for sample, group in zip(samples, groups):
        response_token_ids = [all_generations[i] for i in group]
        finish_reasons = [all_finish_reasons[i] for i in group]
        response_strs = tokenizer.batch_decode(
            response_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        # len(GENERATIONS_PER_SAMPLE)
        # for generation_str in response_strs:
        #     print(f"generation_str is {generation_str}")
        reward_and_metrics = [
            compute_reward(generation_str, sample) for generation_str in response_strs
        ]
        rewards, reward_metrics = zip(*reward_and_metrics)
        # print(f"rewards is {rewards}")
        rewards = np.array(rewards)
        response_advantages = (rewards - np.mean(rewards)) / (rewards.std() + 1e-4)
        response_advantages = [
            [advantage] * len(response_token_id)
            for advantage, response_token_id in zip(
                response_advantages, response_token_ids
            )
        ]
        all_advantages.extend(response_advantages)
        all_query_token_ids.extend([sample["input_ids"]] * generations_per_sample)
        all_response_token_ids.extend(response_token_ids)

        stats["rewards"].extend(rewards.tolist())
        stats["response_length"].extend([len(x) for x in all_response_token_ids])
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        for m in reward_metrics:
            for k, v in m.items():
                stats.setdefault(k, []).append(v)

    # TODO: Should we make this a dataclass?
    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_response_token_ids,
        "all_advantages": all_advantages,
    }
    return episodes, stats


def load_model_into_vllm(model: PreTrainedModel, llm: LLM) -> None:
    """
    Load a HuggingFace model into a VLLM LLM.
    """
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
        model.state_dict().items()
    )


def evaluate_on_test_set(
    inference_engine: LLM,
    test_dataset: Dataset,
    tokenizer: AutoTokenizer,
    eval_sampling_params: SamplingParams,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate the current policy model used by inference engine on the test set.
    """
    generations = inference_engine.generate(
        prompt_token_ids=test_dataset["input_ids"], sampling_params=eval_sampling_params
    )
    metrics = {
        "response_length": [],
        "rewards": [],
        "non_stop_rate": [],
    }
    all_query_tokens = []
    all_response_tokens = []

    # Assuming n=1 for generations here.
    for i, sample in enumerate(test_dataset):
        response_token_ids = generations[i].outputs[0].token_ids
        finish_reasons = generations[i].outputs[0].finish_reason
        response_str = tokenizer.decode(
            response_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        metrics["response_length"].append(len(response_token_ids))
        all_query_tokens.append(sample["input_ids"])
        all_response_tokens.append(response_token_ids)
        metrics["rewards"].append(compute_reward(response_str, sample)[0])
        metrics["non_stop_rate"].append(finish_reasons != "stop")

    episodes = {
        "all_query_token_ids": all_query_tokens,
        "all_response_token_ids": all_response_tokens,
    }
    return episodes, metrics


def dump_training_metrics(
    training_metrics: Dict[str, float],
    iteration: int,
    metrics_file: Path,
):
    with open(metrics_file, "a") as f:
        training_metrics["iteration"] = iteration
        f.write(json.dumps(training_metrics) + "\n")


def dump_episodes(
    episodes: Dict[str, Any],
    episode_stats: Dict[str, Any],
    exp_dir: Path,
    tokenizer: AutoTokenizer,
    iteration: int,
):
    query_texts = tokenizer.batch_decode(
        episodes["all_query_token_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    response_texts = tokenizer.batch_decode(
        episodes["all_response_token_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    episodes_dir = exp_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    data_list = [
        {
            "query_text": query_texts[i],
            "response_text": response_texts[i],
            "response_length": episode_stats["response_length"][i],
            "reward": episode_stats["rewards"][i],
        }
        for i in range(len(query_texts))
    ]
    reward_avg = np.mean(episode_stats["rewards"])
    response_len_avg = np.mean(episode_stats["response_length"])

    with open(episodes_dir / f"eps_{iteration}.jsonl", "w") as f:
        f.write(
            json.dumps({"reward_avg": reward_avg, "response_len_avg": response_len_avg})
            + "\n"
        )
        for data in data_list:
            f.write(json.dumps(data) + "\n")


def get_latest_ckpt(ckpt_dir: Path) -> Tuple[Optional[Path], Optional[int]]:
    """
    Get the latest checkpoint from a directory.
    """
    checkpoints = list(ckpt_dir.glob("ckpt_*"))
    if not checkpoints:
        return None, None

    latest_ckpt = max(checkpoints, key=lambda x: int(x.name.split("_")[1]))
    return latest_ckpt, int(latest_ckpt.name.split("_")[1])
