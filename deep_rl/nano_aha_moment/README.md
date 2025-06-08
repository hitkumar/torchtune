# Using RL to Reproduce Deepseek-R1

## Results
- Baseline Nano_aha_moment achieves **1.608** on the eval set
- After 1000 iterations, I can get **1.43**
- After 2000 iterations, I can get **1.536**

## Analysis
- Difference is likely because Nano_aha_moment uses deepspeed, while I use native PyTorch.
- Added TRL based training, but will rely on verifiers for further exploration in this direction.

## Tests
- Adding using AI assistance
- Run the tests using the following command:
  ```bash
  python -m pytest tests/test_reward_functions.py -c /dev/null
  ```
