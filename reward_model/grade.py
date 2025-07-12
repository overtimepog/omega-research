import json
import numpy as np
import pandas as pd
from typing import List, Dict
import re
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm.backend import score_abstracts_with_vllm, score_abstracts_with_api


def evaluate_and_compare(data: List[Dict], model_results: List[Dict]) -> Dict:
    results = {
        "model_scores": [res["score"] for res in model_results],
        "avg_ratings": [res["avg_rating"] for res in model_results],
        "evaluations": [res["evaluation"] for res in model_results],
        "abstracts": [res["abstract"] for res in model_results],
        "differences": [],
        "mae": 0.0,
        "mse": 0.0,
        "accuracy": 0.0  # New metric for accuracy
    }

    # Calculate differences and labels
    valid_differences = []
    true_labels = []
    pred_labels = []
    for ms, ar in zip(results["model_scores"], results["avg_ratings"]):
        # Calculate difference
        diff = abs(ms - ar) if ms >= 0 else -1
        valid_differences.append(diff)

        # Assign labels: positive (1) if score > 5.5, negative (0) otherwise
        # Only include valid scores for accuracy calculation
        if ms >= 0:
            true_label = 1 if ar > 5.5 else 0
            pred_label = 1 if ms > 5.5 else 0
            true_labels.append(true_label)
            pred_labels.append(pred_label)

    # Calculate metrics
    results["differences"] = valid_differences
    valid_diffs = [d for d in valid_differences if d >= 0]
    results["mae"] = np.mean(valid_diffs) if valid_diffs else 0.0
    results["mse"] = np.mean([d ** 2 for d in valid_diffs]) if valid_diffs else 0.0
    
    # Calculate accuracy: proportion of matching labels
    correct_predictions = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    results["accuracy"] = correct_predictions / len(true_labels) if true_labels else 0.0

    return results

def print_results(results: Dict):
    print(f"{'Index':<6} {'Model Score':<12} {'Avg Rating':<12} {'Difference':<12} {'Evaluation':<50}")
    print("-" * 100)
    for i in range(len(results["model_scores"])):
        eval_snippet = results["evaluations"][i][:47] + "..." if len(results["evaluations"][i]) > 47 else results["evaluations"][i]
        diff = results["differences"][i] if results["differences"][i] >= 0 else "N/A"
        print(f"{i+1:<6} {results['model_scores'][i]:<12.2f} {results['avg_ratings'][i]:<12.2f} {diff:<12} {eval_snippet:<50}")
    print("\nSummary Statistics (excluding invalid scores):")
    print(f"Mean Absolute Error (MAE): {results['mae']:.2f}")
    print(f"Mean Squared Error (MSE): {results['mse']:.2f}")
    print(f"Prediction Accuracy: {results['accuracy']:.2%}")  # Display accuracy as percentage

def main():

    try:
        data = load_dataset('json', data_files='/data/zhuotaodeng/yzj/alpha-research/data/iclr2025_eval_100.json', split='train')
        data = [dict(item) for item in data]  # Convert to List[Dict]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    model_results = score_abstracts_with_vllm(data, model_name="/data/zhuotaodeng/yzj/download_from_modelscope/Qwen/Qwen3-8B")
    # model_results = score_abstracts_with_vllm(data, '/data/zhuotaodeng/yzj/alpha-research/model/qwen25_grm_iclr_boxed/checkpoint-120')
    # model_results = score_abstracts_with_api(data, '/data/zhuotaodeng/yzj/alpha-research/idea-eval/results.jsonl')

    results = evaluate_and_compare(data, model_results)

    print_results(results)

    with open("vllm_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to 'vllm_evaluation_results.json'")

    df = pd.DataFrame({
        "Abstract": results["abstracts"],
        "Model_Score": results["model_scores"],
        "Avg_Rating": results["avg_ratings"],
        "Difference": results["differences"],
        "Evaluation": results["evaluations"]
    })
    df.to_csv("vllm_evaluation_results.csv", index=False)
    print("Results also saved to 'vllm_evaluation_results.csv'")


if __name__ == "__main__":
    main()