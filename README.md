
<div align="center">

# AlphaResearch
<img src="./assets/overview.png" width="800" alt="alpha-research" />

</div>

## TODO:

- Train a Reward Model (Done)
- Run the system iteratively to get new algorithm (Need to find some problems)
- Compare with AlphaEvolve


## EvolveAgent

We use [OpenEvolve](https://github.com/codelion/openevolve) as our evolutionary agent.

## Reward Model

We train Qwen2.5-7B-Instruct with ICLR(2017-2024) papers as our reward model.

- Train Dataset: Abstract and Review Score of  ICLR 2017-2024 papers  (24,445 in total) (knowledge cut-off date: Dec, 2023)

- Evaluation Dataset: Abstract and Review Score of 100 ICLR 2025 papers
(ICLR2025 Rebuttal started at Dec, 2024)

- Metric: positive score (>5.5), negative score(<=5.5), binary classification

### Results

| Model | Released Date (Knowledge Cutoff) | Accuracy (Binary) |
| --- | --- | --- |
| Deepseek-v3-0324 | Mar, 2025 (potential leakage) | 39.0% |
| Qwen3-8B | May, 2025 (potential leakage) | 60.0% |
| Qwen2.5-7B-Instruct | Sep, 2024 | 37.0% |
|              + Fine-tuned  | Sep, 2024  | 72.0% |

