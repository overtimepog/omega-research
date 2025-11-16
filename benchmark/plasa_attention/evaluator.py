"""
PLASA Benchmark Evaluator - November 2025

Evaluates Per-Layer Adaptive Sparse Attention implementations by:
1. Loading the PLASAModel from initial_program.py
2. Training on cosmopedia-v2 (same as exp3) for 1000 steps
3. Measuring validation perplexity and accuracy

Score: 1 / validation_perplexity (higher is better)

Expected Performance (matching exp3):
- Validation Perplexity: ~72-80
- Validation Accuracy: ~50-55%
- Score: ~0.0125-0.0139

Dataset: cosmopedia-v2 (HuggingFaceTB/smollm-corpus)
Tokenizer: SmolLM-135M
"""

import sys
import os
import json
import importlib.util
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List
import pickle

# Add path to evolve_agent for metric normalization utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evolve_agent.utils.metrics_utils import normalize_metric
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TextTokenDataset(Dataset):
    """Token dataset with overlapping windows (same as exp3)"""
    def __init__(self, tokens: List[int], seq_len: int = 128, window_indices: List[int] = None):
        self.tokens = tokens
        self.seq_len = seq_len
        # If window_indices is provided, use those specific windows
        # Otherwise, use all possible windows (original behavior)
        if window_indices is not None:
            self.window_indices = window_indices
        else:
            self.window_indices = list(range(max(0, len(tokens) - seq_len)))

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        # Get the actual window start position
        window_start = self.window_indices[idx]
        x = torch.tensor(self.tokens[window_start:window_start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[window_start + 1:window_start + self.seq_len + 1], dtype=torch.long)
        return x, y


def load_cosmopedia_data(num_documents: int = 1000, max_tokens: int = 2_000_000, cache_dir: str = "data_cache"):
    """
    Load cosmopedia-v2 data with SmolLM tokenizer (exactly matching exp3).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/cosmopedia_{num_documents}_{max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"Loading cached cosmopedia data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        vocab_size = tokenizer.vocab_size

        print(f"Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return tokens, vocab_size, tokenizer

    print(f"Processing cosmopedia-v2 data (will cache for future use)")

    # Load tokenizer (same as exp3)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (same as exp3)
    print("Loading cosmopedia-v2 dataset...")
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(encoded)

    tokens = all_tokens[:max_tokens]
    print(f"Using {len(tokens):,} tokens")
    vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"Cached data to {cache_file}")
    return tokens, vocab_size, tokenizer


def train_model(model, train_loader, optimizer, device, max_steps=1000):
    """Train the model for max_steps"""
    model.train()
    total_loss = 0
    total_tokens = 0
    step = 0

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            # Handle tuple output from dataset (x, y)
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
            step += 1

            if step % 50 == 0:  # Log every 50 steps (same as exp3)
                avg_loss = total_loss / total_tokens
                print(f"  Step {step}/{max_steps}, Loss: {avg_loss:.4f}")

    return total_loss / total_tokens if total_tokens > 0 else 0


def evaluate_model(model, val_loader, device, max_batches=100):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            # Handle tuple output from dataset (x, y)
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            shift_preds = predictions[..., :-1].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            correct = (shift_preds == shift_labels).sum().item()

            total_loss += loss.item() * input_ids.numel()
            total_correct += correct
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
    }


def evaluate(program_path: str = "initial_program.py") -> Dict:
    """
    Evaluate a PLASA implementation using cosmopedia-v2 (same as exp3).

    Args:
        program_path: Path to the PLASA implementation

    Returns:
        Dict with score (1/perplexity) or error
    """
    try:
        # Set all random seeds for reproducibility (same as exp3)
        set_seed(42)
        # Load the program
        spec = importlib.util.spec_from_file_location("plasa_program", program_path)
        plasa_module = importlib.util.module_from_spec(spec)
        sys.modules["plasa_program"] = plasa_module
        spec.loader.exec_module(plasa_module)

        # Extract PLASAModel
        PLASAModel = getattr(plasa_module, 'PLASAModel', None)
        if PLASAModel is None:
            print("Error: PLASAModel not found in initial_program.py")
            return {"error": -1.0}

        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load cosmopedia-v2 data (same as exp3)
        print("Loading cosmopedia-v2...")
        tokens, vocab_size, tokenizer = load_cosmopedia_data(
            num_documents=1000,
            max_tokens=2_000_000
        )

        # Create datasets (same as exp3)
        seq_len = 128
        dataset = TextTokenDataset(tokens, seq_len=seq_len)

        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # Create model (same config as exp3)
        print("Creating PLASA model...")
        model = PLASAModel(
            vocab_size=vocab_size,
            hidden_size=128,
            n_layers=4,
            n_heads=4,
            num_kv_heads=2,  # For compatibility
            head_dim=32,  # For compatibility
            intermediate_size=512,
            max_seq_len=seq_len,
            dropout=0.1,
            rms_norm_eps=1e-6,
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")

        # Training (same as exp3)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

        print("\nTraining for 1000 steps...")
        train_loss = train_model(model, train_loader, optimizer, device, max_steps=1000)

        print("\nEvaluating...")
        val_metrics = evaluate_model(model, val_loader, device, max_batches=100)

        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
        print(f"  Val Perplexity: {val_metrics['perplexity']:.2f}")

        # Score: 1 / perplexity (higher is better)
        # Cap extreme perplexity values before computing score
        perplexity = min(val_metrics['perplexity'], 10000)
        score = 1.0 / perplexity if perplexity > 0 else 0.0

        # Combined score: weighted combination of normalized metrics (higher is better)
        # Based on 2025 NAS best practices: normalize all components to [0,1] with
        # proper direction handling (minimize/maximize) before weighted combination
        #
        # Normalization approach:
        # - perplexity: log-transform + min-max to [0,1], then invert (lower perp = higher score)
        # - accuracy: already in [0,1], use directly
        # - val_loss: min-max to [0,1], then invert (lower loss = higher score)
        #
        # Weights: perplexity (60%), accuracy (30%), val_loss (10%)

        # Normalize each component using metric-aware normalization
        norm_perplexity = normalize_metric(val_metrics['perplexity'], 'perplexity')
        norm_accuracy = normalize_metric(val_metrics['accuracy'], 'accuracy')
        norm_val_loss = normalize_metric(val_metrics['loss'], 'val_loss')

        combined_score = (
            0.6 * norm_perplexity +  # Normalized & inverted: lower perplexity = higher score
            0.3 * norm_accuracy +     # Normalized: higher accuracy = higher score
            0.1 * norm_val_loss       # Normalized & inverted: lower val_loss = higher score
        )

        return {
            "score": float(score),
            "perplexity": float(val_metrics['perplexity']),
            "accuracy": float(val_metrics['accuracy']),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics['loss']),
            "combined_score": float(combined_score),
        }

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error during evaluation: {e}")
        print(error_traceback)

        # Return structured error with full traceback for bug fixer
        # Research: Based on RepairAgent (ICSE 2025) structured error approach
        return {
            "error": -1.0,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": error_traceback,
            "failure_stage": "evaluation"
        }


if __name__ == "__main__":
    try:
        default_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    except:
        default_path = "initial_program.py"

    target = sys.argv[1] if len(sys.argv) > 1 else default_path
    result = evaluate(target)
    print(json.dumps(result, ensure_ascii=False))
