import argparse

import torch
from safetensors import safe_open

parser = argparse.ArgumentParser(description="Compare two safetensors")
parser.add_argument("--tensor1", type=str, help="First tensor to compare")
parser.add_argument("--tensor2", type=str, help="Second tensor to compare")

args = parser.parse_args()

tensor1 = {}
tensor2 = {}

with safe_open(args.tensor1, framework="pt", device="cpu") as f:
    print("Loading tensor1")
    for key in f.keys():
        tensor1[key] = f.get_tensor(key)

with safe_open(args.tensor2, framework="pt", device="cpu") as f:
    print("Loading tensor2")
    for key in f.keys():
        tensor2[key] = f.get_tensor(key)

# Check if the keys are the same
if tensor1.keys() != tensor2.keys():
    print("Keys are not the same")
    print("First tensor keys: ", tensor1.keys())
    print("Second tensor keys: ", tensor2.keys())
    exit(1)

# Check if the tensors are the same
num_same, num_diff = 0, 0
for key in tensor1.keys():
    if not torch.all(torch.eq(tensor1[key], tensor2[key])):
        print("Tensors are not the same")
        print("Key: ", key)
        print("First tensor: ", tensor1[key])
        print("Second tensor: ", tensor2[key])
        num_diff += 1
    else:
        print(f"Tensor {key} is the same")
        num_same += 1

print(f"Tensors Num same: {num_same}, num diff: {num_diff}")
