import re
import os
import json
import time
from typing import List, Dict, Optional
from pathlib import Path

import asyncio
import aiofiles
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from datasets import load_dataset

from evolve_agent.config import RewardModelConfig
from evolve_agent.reward_model import RewardModel

async def test_reward_model():
    # Sample data for testing
    sample_data = [
        {
            "title": "A Novel Approach to Quantum Computing",
            "abstract": "This proposal introduces a new quantum algorithm that enhances computational efficiency by leveraging entangled states in a scalable architecture. The approach is validated through simulations showing a 20% improvement over existing methods.",
            "gt_score": 7.5
        },
        {
            "title": "AI-Driven Climate Modeling",
            "abstract": "We propose an AI-based framework for improving climate predictions using deep learning to integrate heterogeneous environmental data. Preliminary results demonstrate enhanced accuracy in long-term forecasts.",
            "gt_score": 8.0
        }
    ]

    # Test with vLLM model (commented out because it requires a local model and GPU)

    # try:
    #     vllm_model = RewardModel(
    #         model_type="vllm",
    #         model_name="/data/zhuotaodeng/yzj/alpha_research_model/qwen25_grm_iclr_boxed/checkpoint-180",
    #         jsonl_file="vllm_results.jsonl"
    #     )
    #     vllm_results = await vllm_model.score_research_proposal(sample_data)
    #     print("vLLM Results:")
    #     for result in vllm_results:
    #         print(f"Title: {result['title']}, Score: {result['score']}, Evaluation: {result['evaluation'][:50]}...")
    # except Exception as e:
    #     print(f"vLLM test failed: {e}")


    # Test with API model (requires valid API key and base URL)
    try:
        # Replace with your actual API key and base URL
        api_key = "sk-2c3f1f58031b4b86afdb6a8192ea02e2"
        base_url = "https://api.deepseek.com"

        config = RewardModelConfig(
            model_type="api",
            model_name="deepseek-chat",
            api_key=api_key,
            base_url=base_url,
            jsonl_file="api_results.jsonl",
            max_retries=3,
            retry_delay=1
        ) 
        
        api_model = RewardModel(config)
        api_results = await api_model.score_research_proposal(sample_data)
        print("API Results:")
        for result in api_results:
            print(f"Title: {result['title']}, Score: {result['score']}, Evaluation: {result['evaluation'][:50]}...")
    except Exception as e:
        print(f"API test failed: {e}")

# Run the async test
asyncio.run(test_reward_model())