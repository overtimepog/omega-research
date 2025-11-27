# Toxic Trait Tracking System for Evolutionary Code Optimization

## Problem Statement

The current AlphaResearch-based evolution system can waste computational resources by repeatedly exploring similar failed approaches. When an evolutionary branch produces code that performs significantly worse than expected (< 85% of parent performance), the system should:

1. **Recognize the failure pattern** as a "toxic trait"
2. **Prevent breeding** with toxic programs
3. **Remember failures** across evolution runs
4. **Inform the LLM** about previously failed approaches during proposal generation

## User Requirements

> "All ideas should be able to get within 85% of the score of the original, this should be able to be set in config, if they only get 84% or lower of the way, the idea is to be treated as a toxic trait meaning do not breed with it, it should be saved as well for future, a new json of failures should be stored in that experiments benchmark file, and be provided to the llm on every idea generate call so it knows whats already been tried and failed"

### Core Requirements

1. **Configurable threshold**: Set minimum acceptable performance ratio (default: 0.85 = 85%)
2. **Toxic trait marking**: Programs below threshold are flagged as toxic
3. **Breeding exclusion**: Toxic programs excluded from parent/inspiration sampling
4. **Persistent storage**: JSON file tracks failures per benchmark
5. **LLM awareness**: Failure history included in proposal generation prompts

## Research Context (November 2025)

### Negative Selection in Evolutionary Computation

From [ACM GECCO 2025 research](https://dl.acm.org/doi/10.1145/3583133.3590709) on selection methods:
- Performance ranking of selection methods depends on settings
- Down-sampling combined with selection improves efficiency
- **Key insight**: Active filtering of poor performers prevents population pollution

### Tabu Search and Memory Mechanisms

From [quality diversity optimization research](https://quality-diversity.github.io/papers.html):
- Tabu search uses **short-term memory** with lists of recently failed solutions
- **Long-term memory** with visited regions lists for diversification
- Memory prevents revisiting failed solution neighborhoods

### AlphaResearch Failure Analysis

From [AlphaResearch paper (2511.08522)](https://arxiv.org/pdf/2511.08522):
- 6/8 problems showed failure modes in autonomous discovery
- **Key insight**: Systematic failure analysis informs future attempts
- Execution-based verification alone insufficient without failure learning

### LLM Self-Debugging and Feedback

From [2025 LLM code generation research](https://arxiv.org/abs/2506.23034):
- Self-Debug uses few-shot examples of failures
- Self-Edit incorporates execution feedback to avoid repeating errors
- **Key insight**: LLMs benefit from vulnerability hints and failure demonstrations

## Technical Approach

### Components to Build

1. **FailureTracker class** (`evolve_agent/failure_tracker.py`)
   - Track toxic programs with metadata (proposal, metrics, reason)
   - JSON persistence per benchmark
   - Query interface for failure history

2. **Config extensions** (`evolve_agent/config.py`)
   - `toxic_trait_threshold: float = 0.85`
   - `enable_toxic_trait_tracking: bool = True`
   - `toxic_trait_comparison_metric: str = "combined_score"`
   - `failure_history_path: Optional[str]` (defaults to benchmark dir)

3. **Database sampling filters** (`evolve_agent/database.py`)
   - Exclude toxic programs from parent sampling
   - Exclude toxic programs from inspiration sampling
   - Log toxic program statistics

4. **Evaluation logic** (`evolve_agent/controller.py`)
   - Compare child performance to parent
   - Mark as toxic if below threshold
   - Store failure with context

5. **Prompt integration** (`evolve_agent/prompt/sampler.py`)
   - Include failure history in proposal prompts
   - Format as "Failed Approaches" section
   - Limit to most recent/relevant failures

## Success Criteria

- [x] Toxic programs cannot be selected as parents
- [x] Toxic programs excluded from inspirations
- [x] Failures persist across runs in JSON format
- [x] LLM receives failure history during proposal generation
- [x] Threshold configurable in YAML config
- [x] System works with existing MAP-Elites + Islands architecture

## References

- [Negative Selection Algorithms](https://ieeexplore.ieee.org/document/9546626/) - IEEE review of NSA mechanisms
- [AlphaResearch: Accelerating Algorithm Discovery](https://arxiv.org/pdf/2511.08522) - Failure mode analysis
- [LLM-Guided Code Evolution](https://arxiv.org/abs/2506.23034) - Self-debugging with failure feedback
- [Quality Diversity Optimization](https://quality-diversity.github.io/papers.html) - Memory mechanisms
- [Tabu Search Memory Structures](https://en.wikipedia.org/wiki/Tabu_search) - Short and long-term memory
- [Selection Methods in GP (GECCO 2025)](https://dl.acm.org/doi/10.1145/3712255.3734222) - Performance analysis
