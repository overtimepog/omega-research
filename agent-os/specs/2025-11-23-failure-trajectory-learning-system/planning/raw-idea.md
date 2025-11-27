# Failure Trajectory Learning System - Raw Idea

**Date Created:** 2025-11-23
**Project:** Omega-Research
**Status:** Planning Phase

## Feature Description

Implement a comprehensive failure trajectory learning system that:

### 1. Outcome Classification System
Classify evaluation results into typed outcomes (HARD_FAIL, SOFT_FAIL, NEAR_MISS, SUCCESS, SOTA) using problem-specific normalized scores and thresholds

### 2. Failure Point Detection
Mark low-but-working scores as explicit "failure points" in evolutionary trajectories when they:
- Produce valid code that runs successfully
- Score worse than parent solutions
- Fall below problem-specific "soft fail" thresholds

### 3. Experience Store
Build a persistent, problem-specific JSONL log capturing:
- Full trajectory steps (code, ideas, scores, outcomes)
- Failure point annotations
- Parent-child lineage relationships
- Rich metadata (RM scores, eval logs, generation index)

### 4. Local Prior Model
Train a small, per-problem ML model that:
- Learns from accumulated experience what patterns lead to success vs failure
- Uses text embeddings + scalar features to predict outcome likelihood
- Guides parent selection away from known failure basins
- Enables epsilon-greedy exploration to avoid over-exploitation

### 5. Failure Memory Integration
Inject learned anti-patterns into LLM prompts:
- Summarize concrete failure examples from experience logs
- Construct "what not to do" guidance for idea generation
- Make prompts problem-specific and data-driven

### 6. OpenEvolve Integration
Hook into existing AlphaResearch/OpenEvolve workflow:
- Wrap evaluation callbacks to classify outcomes
- Modify parent selection with prior-based filtering
- Enhance mutation prompts with failure context
- Maintain backward compatibility with existing benchmarks

## Current Codebase Context

- Uses OpenEvolve as evolutionary agent
- Has 8+ benchmark problems (circle packing, autocorrelation, MSTD, etc.)
- Uses AlphaResearch reward model (peer-review-based RM)
- Entry point: run.py, core logic in evolve_agent/
- Score baselines and directions in compute.py

## Target Behavior

After 1000+ evaluations per problem, the system should spend less time re-exploring known failure regions and more time refining near-misses and exploring novel solution spaces. The evolver builds its own domain-specific knowledge about what works for each specific benchmark.

## Expected Benefits

- Reduced waste of compute on known failure patterns
- Faster convergence to high-quality solutions
- Problem-specific learning accumulation
- Data-driven prompt engineering based on actual failures
- More efficient exploration of solution space
