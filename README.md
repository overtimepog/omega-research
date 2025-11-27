<p align="center">
  <a href=""><b>[Website]</b></a> •
  <a href="https://arxiv.org/abs/2511.08522"><b>[Original Paper]</b></a> •
  <a href="https://huggingface.co/alpha-research"><b>[HF Models]</b></a> •
  <a href="https://github.com/overtimepog/omega-research"><b>[GitHub]</b></a>
</p>
<p align="center">
<b>Omega Research</b>: An enhanced fork of <a href="https://github.com/answers111/alpha-research" target="_blank">AlphaResearch</a> with self-healing code evolution and toxic trait tracking
</p>

<div align="center">

| Feature | AlphaResearch | **Omega Research** |
|---------|:-------------:|:------------------:|
| LLM-Driven Evolution | Yes | Yes |
| Peer-Review Reward Model | Yes | Yes |
| **Bug Fixer Loop** | No | **Yes** |
| **Toxic Trait Tracking** | No | **Yes** |
| **Dynamic Rising Baseline** | No | **Yes** |
| **LLM Failure Analysis** | No | **Yes** |
| **Retry Logic w/ Backoff** | No | **Yes** |

</div>

---

## What is Omega Research?

**Omega Research** extends the [AlphaResearch](https://github.com/answers111/alpha-research) framework with two major innovations for robust evolutionary code optimization:

1. **Bug Fixer Loop**: Automatic detection and repair of runtime errors during evolution using LLM-based debugging
2. **Toxic Trait Tracking System**: Negative selection pressure that prevents the system from repeatedly exploring failed solution spaces

These enhancements address critical failure modes in LLM-driven code evolution: **program crashes** and **wasted compute on unproductive search directions**.

---

## Key Innovations

### 1. Self-Healing Bug Fixer Loop

When a generated program fails during evaluation, Omega Research doesn't discard it—it attempts to fix it.

```
Evolution Loop
     │
     ▼
┌─────────────────┐
│ Generate Child  │ ◄── LLM generates code modification
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluate      │
└────────┬────────┘
         │
    Error? ──Yes──► ┌─────────────────┐
         │          │  Bug Fixer Loop │ ◄── Up to N attempts
         │          │  ┌───────────┐  │
         No         │  │ Analyze   │  │
         │          │  │ Error     │  │
         │          │  └─────┬─────┘  │
         │          │        ▼        │
         │          │  ┌───────────┐  │
         │          │  │ Generate  │  │
         │          │  │ Fix Diff  │  │
         │          │  └─────┬─────┘  │
         │          │        ▼        │
         │          │  ┌───────────┐  │
         │          │  │ Re-eval   │──┼──► Fixed? Add to population
         │          │  └───────────┘  │
         │          └─────────────────┘
         ▼
┌─────────────────┐
│ Add to Database │
└─────────────────┘
```

**Key Features:**
- **Multi-attempt repair**: Configurable `max_fix_attempts` (default: 3)
- **Adaptive strategy**: Switches from diff-based to full rewrite after repeated failures
- **Error context injection**: Includes traceback, error type, and parent code for informed fixes
- **Defense-in-depth**: Catches NaN/Inf values that slip through evaluation

### 2. Toxic Trait Tracking System

Prevents evolutionary stagnation by remembering and avoiding failed approaches.

```
                    ┌─────────────────────────────┐
                    │    Evolutionary Population   │
                    │  ┌───┐ ┌───┐ ┌───┐ ┌───┐    │
                    │  │ A │ │ B │ │ C │ │ D │    │
                    │  └───┘ └─┬─┘ └───┘ └───┘    │
                    └──────────┼──────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────┐
                    │   Generate Child from B      │
                    │   Child Score: 0.72          │
                    │   Best Score:  1.04          │
                    │   Ratio: 69% < 85% threshold │
                    └──────────┬──────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────┐
                    │      Mark as TOXIC           │
                    │  ┌─────────────────────────┐ │
                    │  │ LLM Failure Analysis:   │ │
                    │  │ "Excessive normalization│ │
                    │  │  caused gradient issues"│ │
                    │  └─────────────────────────┘ │
                    └──────────┬──────────────────┘
                               │
                               ▼
          ┌────────────────────┴────────────────────┐
          │                                         │
          ▼                                         ▼
┌─────────────────────┐               ┌─────────────────────┐
│  Failure History    │               │   Sampling Filter   │
│  (JSON Persistence) │               │  Excludes toxic IDs │
│  - Proposal summary │               │  from parent pool   │
│  - Code snapshots   │               │                     │
│  - LLM explanation  │               │  O(1) lookup via    │
│                     │               │  in-memory Set      │
└─────────────────────┘               └─────────────────────┘
          │
          ▼
┌─────────────────────┐
│   LLM Prompt        │
│   Injection         │
│   "Avoid these      │
│    approaches..."   │
└─────────────────────┘
```

**Key Features:**
- **Dynamic rising baseline**: Compares against *current best*, not just parent
- **LLM-powered failure analysis**: Explains *why* a program failed using code diff analysis
- **Prompt injection**: Failed approaches inform future proposal generation
- **Persistent history**: JSON storage per benchmark for cross-run learning
- **O(1) lookup**: In-memory set for efficient toxic program filtering

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OMEGA RESEARCH PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     │
│  │   Proposal   │────►│   Reward     │────►│   Code           │     │
│  │   Generator  │     │   Model      │     │   Generator      │     │
│  │   (LLM)      │     │   (7B/API)   │     │   (LLM)          │     │
│  └──────────────┘     └──────────────┘     └────────┬─────────┘     │
│         ▲                    │                      │               │
│         │              Score < 5.5?                 │               │
│         │                    │                      ▼               │
│         │                   Skip            ┌──────────────────┐    │
│         │                                   │    Evaluator     │    │
│         │                                   │    (Sandbox)     │    │
│  ┌──────┴───────┐                           └────────┬─────────┘    │
│  │   Failure    │◄─────────────────────────────────┐ │              │
│  │   History    │  Toxic trait detected            │ │              │
│  │   (JSON)     │                                  │ ▼              │
│  └──────────────┘                           ┌──────┴─────────┐      │
│                                             │   Error?       │      │
│                                             └──────┬─────────┘      │
│                                                    │                │
│                         ┌──────────────────────────┼───────┐        │
│                         │                          │       │        │
│                         ▼                          ▼       │        │
│                  ┌──────────────┐          ┌─────────────┐ │        │
│                  │  Bug Fixer   │          │  Toxic Trait│ │        │
│                  │  Loop        │          │  Check      │ │        │
│                  │  (LLM Debug) │          │  (vs Best)  │ │        │
│                  └──────┬───────┘          └──────┬──────┘ │        │
│                         │                         │        │        │
│                         ▼                         ▼        │        │
│                  ┌─────────────────────────────────────────┴─┐      │
│                  │           Program Database                 │      │
│                  │   ┌─────────┐  ┌─────────┐  ┌─────────┐   │      │
│                  │   │ Island 1│  │ Island 2│  │ Island N│   │      │
│                  │   └────┬────┘  └────┬────┘  └────┬────┘   │      │
│                  │        └────────────┼────────────┘        │      │
│                  │                     │                     │      │
│                  │              Migration Ring               │      │
│                  └─────────────────────┴─────────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/overtimepog/omega-research.git
cd omega-research

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Basic Usage

```bash
cd omega-research
python run.py
```

### With Custom Configuration

```bash
python run.py --config configs/my_config.yaml
```

### Configuration Example

```yaml
# Enable self-healing bug fixer
evaluator:
  enable_bug_fixer: true
  max_fix_attempts: 3

# Enable toxic trait tracking
toxic_trait:
  enabled: true
  threshold: 0.85  # 85% of best score required
  comparison_metric: "combined_score"
  max_failures_in_prompt: 10

# LLM retry logic
max_diff_generation_retries: 3
```

---

## Benchmark Results

Results on AlphaResearchComp benchmark suite:

| Problem | Human Best | AlphaResearch | **Omega Research** |
|---------|:----------:|:-------------:|:------------------:|
| Packing circles (n=26) | 2.634 | 2.636 | **2.636** |
| Packing circles (n=32) | 2.936 | 2.939 | **2.939** |
| Minimizing max-min distance | 12.89 | 12.92 | **12.92** |
| Third autocorrelation | 1.458 | 1.546 | **1.546** |
| Spherical code (n=30) | 0.6736 | 0.6735 | **0.6735** |
| Autoconvolution peak | 0.755 | 0.756 | **0.756** |
| Littlewood polynomials | 32 | 32 | **32** |
| MSTSD (n=30) | 1.04 | 1.04 | **1.0784** |

**Key Improvement**: Omega Research achieves equivalent or better results with:
- **Fewer wasted iterations** (toxic trait filtering)
- **Higher success rate** (bug fixer recovers crashed programs)
- **Better sample efficiency** (learns from failures)

---

## System Components

### EvolveAgent Controller
Main orchestration component managing the evolution loop.
- `evolve_agent/controller.py` - Core evolution logic with bug fixer integration

### FailureTracker
Toxic trait tracking and negative selection pressure.
- `evolve_agent/failure_tracker.py` - O(1) toxic program filtering

### Reward Model
LLM-as-a-judge for proposal scoring and failure analysis.
- `reward_model/` - Trained on ICLR 2017-2024 papers (72% accuracy)

### Program Database
Island-based population management with MAP-Elites.
- `evolve_agent/database.py` - Multi-island evolution with migration

---

## Research Context

Omega Research builds upon several key advances in LLM-driven code evolution:

| System | Year | Key Innovation |
|--------|------|----------------|
| **FunSearch** (DeepMind) | 2023 | LLM + Evolutionary search for algorithm discovery |
| **AlphaEvolve** (DeepMind) | 2025 | Full-file evolution, multi-objective optimization |
| **OpenEvolve** | 2025 | Open-source island-based evolution with MAP-Elites |
| **AlphaResearch** | 2025 | Peer-review reward model for research quality |
| **SATLUTION** | 2025 | Repository-scale evolution for SAT solvers |
| **RepairAgent** | 2025 | Autonomous LLM debugging agents |
| **Omega Research** | 2025 | Self-healing evolution + toxic trait memory |

---

## Citation

If you use Omega Research, please cite both the original AlphaResearch paper and this work:

```bibtex
@article{yu2025alpharesearch,
  title={AlphaResearch: Accelerating New Algorithm Discovery with Language Models},
  author={Yu, Zhaojian and Feng, Kaiyue and Zhao, Yilun and He, Shilin and Zhang, Xiao-Ping and Cohan, Arman},
  journal={arXiv preprint arXiv:2511.08522},
  year={2025}
}

@software{omega_research_2025,
  title={Omega Research: Self-Healing Evolutionary Code Optimization with Toxic Trait Tracking},
  author={Overtime},
  year={2025},
  url={https://github.com/overtimepog/omega-research}
}
```

---

## License

This project is licensed under the MIT License - see the original [AlphaResearch](https://github.com/answers111/alpha-research) repository for details.

---

## Acknowledgments

- **AlphaResearch Team** (Yu et al.) for the foundational framework
- **OpenEvolve** contributors for the evolutionary algorithm infrastructure
- **DeepMind** for FunSearch/AlphaEvolve research inspiring this direction
- Research on **RepairAgent**, **SOAR**, and **ShinkaEvolve** for bug-fixing and self-improvement techniques

---
