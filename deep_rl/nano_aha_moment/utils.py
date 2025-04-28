from typing import Any, Dict, List, Tuple

import numpy as np

from reward_functions import compute_reward
from transformers import AutoTokenizer


GENERATIONS_PER_SAMPLE = 4


def create_training_episodes(
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, Any]],
    all_generations: List[List[int]],
    all_finish_reasons: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create training episodes from samples and generations.
    """

    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE
    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE))
        for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
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
        rewards = [
            compute_reward(generation_str, sample)[0]
            for generation_str in response_strs
        ]
        rewards = np.array(rewards)
        response_advantages = (rewards - np.mean(rewards)) / (rewards.std() + 1e-6)
        all_advantages.extend(response_advantages)
        all_query_token_ids.extend(sample["input_ids"] * GENERATIONS_PER_SAMPLE)
        all_response_token_ids.extend(response_token_ids)

        stats["rewards"].extend(rewards.tolist())
        stats["response_length"].extend([len(x) for x in all_response_token_ids])
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])

    # TODO: Should we make this a dataclass?
    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_response_token_ids,
        "all_advantages": all_advantages,
    }
    return episodes, stats
