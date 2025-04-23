from vllm import LLM, SamplingParams

trained_model_path = "/home/htkumar/tune_logs/llama3_2_3B/lora/epoch_0"

llm = LLM(
    model=trained_model_path,
    load_format="safetensors",
    kv_cache_dtype="auto",
)


def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


sampling_params = SamplingParams(max_tokens=16, temperature=0.5)

conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Where is Taj Mahal located?"},
    {
        "role": "assistant",
        "content": "The Taj Mahal is located in Agra, India. It is situated on",
    },
    {"role": "user", "content": "In which year was it built?"},
]

outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
print_outputs(outputs)
