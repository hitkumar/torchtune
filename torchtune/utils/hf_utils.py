from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

trained_model_path = "/home/htkumar/tune_logs/llama3_2_3B/lora/epoch_0"
original_model_path = "/home/htkumar/models/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(original_model_path)
tokenizer = AutoTokenizer.from_pretrained(original_model_path)
peft_model = PeftModel.from_pretrained(model, trained_model_path)


def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"inputs: {inputs}")
    outputs = model.generate(**inputs, max_length=20)
    # print(f"outputs: {outputs}")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# print(generate_text(peft_model, tokenizer, "Where is taj mahal located?"))

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=trained_model_path
)

print(generate_text(model, tokenizer, "Where is taj mahal located?"))

# Push trained model to huggingface hub.
# import huggingface_hub

# api = huggingface_hub.HfApi()
# username = huggingface_hub.whoami()["name"]
# print(username)
# repo_name = "torchtune_models"

# # if the repo doesn't exist
# repo_id = huggingface_hub.create_repo(repo_name).repo_id

# # if it already exists
# repo_id = f"{username}/{repo_name}"

# api.upload_folder(
#     folder_path=trained_model_path, repo_id=repo_id, repo_type="model", create_pr=False
# )

# Free memory
del model
del tokenizer
del peft_model
