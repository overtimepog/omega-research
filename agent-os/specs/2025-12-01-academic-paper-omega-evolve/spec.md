# Specification: Academic Paper - Omega Evolve

## Title (Working)

**"Omega Evolve: Accelerating Algorithm Discovery through Negative Selection and Peer-Review Reward Modeling"**

Alternative titles:
- "Learning from Failure: Toxic Trait Tracking for LLM-Guided Program Evolution"
- "Beyond AlphaResearch: Dynamic Baselines and Failure-Driven Learning in Evolutionary Code Optimization"

---

## Goal

Produce a publication-ready academic paper (8-10 pages + appendix) documenting the novel contributions of the Omega Research system, targeting top ML/AI venues (NeurIPS, ICML, ICLR) or evolutionary computation venues (GECCO).

---

## Abstract Draft

Large language models (LLMs) have emerged as powerful mutation operators for evolutionary program synthesis, enabling systems like FunSearch and AlphaEvolve to discover novel algorithms for open mathematical problems. However, existing approaches suffer from wasted computation on failed exploration paths and lack mechanisms to learn from failures. We present **Omega Evolve**, an evolutionary code optimization framework that introduces three key innovations: (1) **Toxic Trait Tracking**, a negative selection mechanism that excludes underperforming programs from breeding using dynamic baseline comparison against the current best solution rather than the immediate parent; (2) **Peer-Review Reward Decoupling**, which scores research proposals independently of program performance using a reward model trained on real peer review data, enabling early filtering of low-quality ideas; and (3) **Failure-Driven Learning**, which records failure patterns with LLM-generated root cause analysis and injects this history into future proposal prompts. On the AlphaResearchComp benchmark of 8 frontier mathematical problems, Omega Evolve achieves competitive results while reducing wasted iterations by [X]% compared to baselines without toxic trait tracking. We release our code, trained reward model, and benchmark results to facilitate reproducible research in LLM-guided algorithm discovery.

---

## Core Contributions

### Contribution 1: Toxic Trait Tracking System

**Novelty**: First system to implement negative selection with dynamic baseline comparison in LLM-guided program evolution.

**Technical Details**:
- Compare child program performance against CURRENT BEST (not parent)
- If `child_score / best_score < threshold` (default 0.85), mark as "toxic"
- Toxic programs excluded from `_sample_parent()` and `_sample_inspirations()`
- O(1) lookup using in-memory set for hot path efficiency
- Threshold rises automatically as better solutions are found

**Why It Matters**:
- Prevents wasted compute on programs that regress significantly
- Creates rising selection pressure as evolution progresses
- Maintains exploration diversity while filtering dead-ends

### Contribution 2: Peer-Review Reward Decoupling

**Novelty**: Separates idea quality assessment from program execution performance.

**Technical Details**:
- Reward model (Qwen2.5-7B) trained on ICLR 2017-2024 peer reviews (24,445 papers)
- Scores proposals 1-10 before code generation
- Threshold filtering: skip code generation if proposal score < 5.5
- 72% accuracy predicting paper acceptance on ICLR 2025 holdout

**Why It Matters**:
- Filters low-quality ideas before expensive LLM code generation
- Mimics real research: good idea + good execution = success
- Reduces API costs by avoiding poor proposals

### Contribution 3: Failure-Driven Learning

**Novelty**: Records failure trajectories with LLM root cause analysis and uses them to guide future generations.

**Technical Details**:
- Store failure records: program_id, parent_id, code, metrics, proposal, failure_reason
- LLM analyzes each failure to generate technical explanation
- Recent failures (last 10) included in proposal generation prompts
- Pattern: "Previously Failed Approaches (avoid these): ..."

**Why It Matters**:
- Prevents system from repeating same mistakes
- Provides contextual guidance to LLM for better proposals
- Builds institutional memory across evolution

---

## Relationship to Original AlphaResearch

**Important**: Omega Research is built on top of the original AlphaResearch codebase (https://github.com/answers111/alpha-research). We extend it with novel contributions while preserving the core peer-review reward system.

### What We Inherit from AlphaResearch:
- OpenEvolve-based evolution loop
- Peer-review reward model (Qwen2.5-7B on ICLR 2017-2024)
- MAP-Elites + Island-based evolution
- AlphaResearchComp benchmark (8 problems)

### What Omega Research Adds (Novel):
1. **Toxic Trait Tracking**: Negative selection with dynamic baseline
2. **Proposal Score Filtering**: Hard threshold at 5.5 to skip bad proposals
3. **Failure Analysis**: LLM explains why programs fail
4. **Failure History Injection**: Recent failures included in prompts
5. **Automated Bug Fixer Loop**: Auto-fixes buggy programs with retry logic (up to 3 attempts)
6. **Automated Change Documentation**: LLM-generated best_solution_changes.md
7. **OpenRouter-Based Architecture**: All LLM calls (generation, scoring, analysis) via OpenRouter API instead of local models

### Code-Level Differences:
| Component | Original AlphaResearch | Omega Research |
|-----------|----------------------|----------------|
| `failure_tracker.py` | Not present | **280 lines (new)** |
| `controller.py` | ~800 lines | +400 lines (toxic tracking + bug fixer) |
| `config.py` | No ToxicTraitConfig | **ToxicTraitConfig added** |
| `database.py` | No toxic filtering | **toxic_programs in sample()** |
| `reward_model.py` | Local vLLM scoring | **OpenRouter API + explain_failure()** |
| `llm/ensemble.py` | Local models | **OpenRouter with weighted model sampling** |

### Infrastructure Differences:
| Aspect | Original AlphaResearch | Omega Research |
|--------|----------------------|----------------|
| **Reward Model** | Local vLLM inference | OpenRouter API (cloud) |
| **Code Generation** | Local or API | OpenRouter API (GPT-5.1-Codex-Mini) |
| **Failure Analysis** | Not present | OpenRouter API |
| **Bug Fixing** | Not present | OpenRouter API with retry logic |
| **Deployment** | Requires local GPU for RM | API-only, no local GPU needed |

---

## Related Work Positioning

### 1. Evolutionary Program Synthesis

| System | Year | Key Innovation | Limitation Omega Addresses |
|--------|------|----------------|---------------------------|
| **FunSearch** | 2023 | LLM as mutation operator | Simple functions only, millions of samples |
| **AlphaEvolve** | 2025 | Multi-file programs, accelerator evaluation | Closed-source, no failure learning |
| **OpenEvolve** | 2025 | Open-source AlphaEvolve alternative | No negative selection, no reward decoupling |
| **ShinkaEvolve** | 2025 | Sample-efficient, novelty rejection | No dynamic baseline, no failure analysis |
| **AlphaResearch** | 2025 | Peer-review reward model | **No toxic trait tracking (we add this)** |

### 2. Reward Modeling for Code

- **CodePRM** (2025): Process reward models using execution feedback
- **μCODE** (2025): Multi-turn code generation with learned verifiers
- **P-GRPO** (2025): Posterior-conditioned rewards to mitigate reward hacking
- **Our work**: Peer-review rewards for research proposals, not just code correctness

### 3. Quality-Diversity in GP

- **MAP-Elites**: Maintains diverse elite solutions across feature space
- **Island-based evolution**: Parallel populations with migration
- **Our work**: Combines QD with toxic trait filtering for negative selection

---

## Methodology Section Structure

### 4.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      OMEGA EVOLVE PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Proposal │───▶│  Reward   │───▶│   Code   │───▶│ Evaluate │ │
│  │Generator │    │  Scoring  │    │Generation│    │ Program  │ │
│  └──────────┘    └───────────┘    └──────────┘    └──────────┘ │
│       ▲              │                               │         │
│       │              │ score < 5.5?                  │         │
│       │              ▼ SKIP                          │         │
│       │         ┌─────────┐                          │         │
│       │         │  Filter │                          │         │
│       │         └─────────┘                          │         │
│       │                                              ▼         │
│       │                                        ┌──────────┐    │
│       │                                        │  Toxic   │    │
│       │                                        │  Check   │    │
│       │                                        └──────────┘    │
│       │                                              │         │
│       │              ┌───────────────────────────────┤         │
│       │              │                               │         │
│       │              ▼                               ▼         │
│       │         toxic=True                      toxic=False    │
│       │              │                               │         │
│       │              ▼                               ▼         │
│       │    ┌──────────────┐                  ┌────────────┐    │
│       │    │   Failure    │                  │  Database  │    │
│       │    │   Tracker    │                  │  (MAP-E)   │    │
│       │    └──────────────┘                  └────────────┘    │
│       │              │                               │         │
│       └──────────────┴───────────────────────────────┘         │
│                      (sampled for next iteration)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Peer-Review Reward Model

**Training Data**: ICLR 2017-2024 papers with review scores
- 24,445 papers total
- Input: Abstract text
- Output: Average review score (1-10)
- Binarized at 5.5 for accept/reject

**Model Architecture**:
- Base: Qwen2.5-7B-Instruct
- Fine-tuned on review prediction task
- JSON output format: `{"score": float, "explanation": str}`

**Integration**:
```python
proposal_score = await reward_model.score_proposal(new_proposal)
if proposal_score < config.proposal_score_threshold:  # 5.5
    logger.info(f"Skipping low-quality proposal (score={proposal_score})")
    continue  # Skip code generation
```

### 4.3 Toxic Trait Tracking

**Algorithm**:
```
Input: child_program, best_program, threshold=0.85
Output: is_toxic (bool)

1. child_score = child_program.metrics[comparison_metric]
2. best_score = best_program.metrics[comparison_metric]
3. performance_ratio = child_score / best_score
4. is_toxic = (performance_ratio < threshold)

IF is_toxic:
    5. failure_reason = LLM_analyze_failure(child, parent)
    6. failure_tracker.add_failure(child, failure_reason)
    7. EXCLUDE child from database
ELSE:
    8. database.add(child)
```

**Key Design Decision**: Compare against BEST, not parent
- Parent comparison: threshold is static
- Best comparison: threshold rises as evolution progresses
- Effect: Increasingly strict selection pressure over time

### 4.4 Failure-Driven Learning

**Failure Record Schema**:
```json
{
  "program_id": "uuid",
  "parent_id": "uuid",
  "timestamp": 1733090400.0,
  "iteration": 42,
  "proposal_summary": "Improved sorting with memory locality",
  "failure_reason": "Loop interchange broke cache coherence",
  "performance_ratio": 0.72,
  "threshold": 0.85
}
```

**Prompt Injection**:
```
=== Previously Failed Approaches (avoid repeating) ===
1. "Improved sorting with memory locality" → Failed: Loop interchange broke cache coherence (achieved 72% of best)
2. "Parallel prefix sum optimization" → Failed: Race condition in accumulator (achieved 68% of best)
...
```

### 4.5 MAP-Elites + Island Evolution

- **Feature Map**: 2D grid (score × complexity)
- **Islands**: 5 parallel populations with migration every 50 iterations
- **Archive**: Elite programs (top 30% by quality)
- **Toxic Filtering**: Applied at sampling time, O(1) lookup

---

## Experimental Design

### 5.1 Benchmarks (AlphaResearchComp)

| Problem | Domain | Metric | Human Best | Direction |
|---------|--------|--------|------------|-----------|
| Packing circles (n=26) | Geometry | Total radius | 2.634 | Higher ↑ |
| Packing circles (n=32) | Geometry | Total radius | 2.936 | Higher ↑ |
| Max-min distance ratio (n=16) | Geometry | Ratio | 12.89 | Lower ↓ |
| Third autocorrelation | Harmonic Analysis | Value | 1.458 | Lower ↓ |
| Spherical code (d=3, n=30) | Geometry | Min angle | 0.6736 | Higher ↑ |
| Autoconvolution peak | Signal Processing | Peak value | 0.755 | Lower ↓ |
| Littlewood polynomials (n=512) | Harmonic Analysis | Supremum | 32 | Higher ↑ |
| MSTD (n=30) | Combinatorics | |A+A|/|A-A| | 1.04 | Higher ↑ |

### 5.2 Ablation Studies

**Ablation 1: Toxic Trait Tracking**
- Condition A: Toxic tracking enabled (threshold=0.85)
- Condition B: Toxic tracking disabled
- Measure: Iterations to convergence, best score achieved, wasted iterations

**Ablation 2: Threshold Sensitivity**
- Thresholds: 0.70, 0.80, 0.85, 0.90, 0.95
- Measure: Trade-off between exploration and exploitation

**Ablation 3: Baseline Comparison Type**
- Condition A: Compare against BEST (dynamic)
- Condition B: Compare against PARENT (static)
- Measure: Convergence speed, final score

**Ablation 4: Proposal Score Filtering**
- Condition A: Filter proposals < 5.5
- Condition B: No filtering
- Measure: Code generation efficiency, final score

### 5.3 Baselines

1. **OpenEvolve** (no toxic tracking, no reward model)
2. **ShinkaEvolve** (novelty rejection, no dynamic baseline)
3. **AlphaResearch** (peer-review rewards, no toxic tracking)
4. **Omega Evolve (ours)** (all features)

### 5.4 Metrics

- **Excel@best**: Percent excess over human best (direction-aware)
- **Iterations to threshold**: Steps to reach 95% of human best
- **Wasted iterations**: Programs marked toxic / total programs
- **Compute efficiency**: Best score per 1000 LLM calls

---

## Results Section Structure

### Table 1: Main Results on AlphaResearchComp

| Problem | Human | OpenEvolve | ShinkaEvolve | AlphaResearch | Omega (Ours) |
|---------|-------|------------|--------------|---------------|--------------|
| Circles (26) | 2.634 | 2.632 | 2.635 | 2.636 | **2.638** |
| Circles (32) | 2.936 | 2.935 | 2.938 | 2.939 | **2.940** |
| ... | ... | ... | ... | ... | ... |

### Table 2: Ablation - Toxic Trait Tracking

| Metric | Disabled | Enabled (0.85) | Improvement |
|--------|----------|----------------|-------------|
| Iterations to 95% | X | Y | -Z% |
| Wasted iterations | A% | B% | -C% |
| Final score | D | E | +F% |

### Figure 1: Convergence Curves

- X-axis: Iteration
- Y-axis: Best score achieved
- Lines: With/without toxic tracking for each benchmark

### Figure 2: Threshold Sensitivity

- X-axis: Threshold (0.7 to 0.95)
- Y-axis: Final score / Wasted iterations (dual y-axis)

---

## Discussion Points

### Limitations

1. **Benchmark dependency**: Results specific to AlphaResearchComp; generalization to other domains unverified
2. **Threshold sensitivity**: Optimal threshold may vary by problem domain
3. **Reward model bias**: Trained on ML papers; may not generalize to all algorithm discovery tasks
4. **Compute requirements**: Still requires significant LLM API calls (thousands per problem)

### Future Work

1. **Adaptive thresholds**: Learn optimal threshold per problem dynamically
2. **Similarity-based rejection**: Use embeddings to detect similar-to-failed proposals
3. **Multi-objective toxic tracking**: Different thresholds for different metrics
4. **Transfer learning**: Apply failure patterns across benchmarks

---

## Writing Checklist

- [ ] Abstract (250 words)
- [ ] Introduction (1.5 pages)
- [ ] Related Work (1.5 pages)
- [ ] Method (2.5 pages)
- [ ] Experiments (1.5 pages)
- [ ] Results (1 page)
- [ ] Discussion (0.5 pages)
- [ ] Conclusion (0.5 pages)
- [ ] References (not counted)
- [ ] Appendix (implementation details, additional results)

---

## Technical Requirements

### Code Artifacts
- Clean, well-documented implementation
- Reproducibility scripts
- Configuration files for all experiments
- Pre-trained reward model weights

### Figures
- System architecture diagram (vector graphics)
- Convergence curves (matplotlib/seaborn)
- Ablation visualizations
- Failure pattern analysis

### Tables
- Main results comparison
- Ablation studies
- Compute efficiency analysis
- Hyperparameter sensitivity

---

## Out of Scope

- Extensive hyperparameter search beyond stated ablations
- New benchmark problem creation
- Theoretical analysis of convergence guarantees
- User studies or qualitative evaluation
- Comparison with non-LLM evolutionary methods (classical GP)
