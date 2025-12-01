# Tasks: Academic Paper - Omega Evolve

## Task Group 1: Paper Infrastructure Setup

- [ ] **1.1 Create paper directory structure**
  - Create `/paper/` directory with subdirectories: `sections/`, `figures/`, `tables/`, `references/`
  - Initialize LaTeX template (NeurIPS 2026 style or arXiv preprint)
  - Set up bibliography file `references.bib`

- [ ] **1.2 Gather existing results from codebase**
  - Extract benchmark results from `benchmark/*/evolve_agent_output/`
  - Collect configuration files used for experiments
  - Document exact git commits for reproducibility

- [ ] **1.3 Create figure templates**
  - System architecture diagram (draw.io or tikz)
  - Convergence curve template (matplotlib)
  - Ablation visualization template

## Task Group 2: Write Introduction Section

- [ ] **2.1 Write opening hook and motivation**
  - Establish the importance of automated algorithm discovery
  - Cite FunSearch, AlphaEvolve success stories
  - Identify the problem: wasted compute on failed explorations

- [ ] **2.2 Present the research gap**
  - No existing system combines negative selection + peer-review rewards + failure learning
  - Current systems don't learn from failures
  - Need for dynamic baseline comparison

- [ ] **2.3 State contributions clearly**
  - Contribution 1: Toxic Trait Tracking with dynamic baseline
  - Contribution 2: Peer-Review Reward Decoupling
  - Contribution 3: Failure-Driven Learning
  - Contribution 4: Open-source release with benchmarks

- [ ] **2.4 Outline paper structure**
  - Brief roadmap of remaining sections

## Task Group 3: Write Related Work Section

- [ ] **3.1 LLM-Guided Program Synthesis subsection**
  - FunSearch (2023): LLM as mutation operator
  - AlphaEvolve (2025): Multi-file programs, accelerator evaluation
  - OpenEvolve (2025): Open-source alternative
  - ShinkaEvolve (2025): Sample efficiency, novelty rejection
  - AlphaResearch (2025): Peer-review rewards

- [ ] **3.2 Reward Modeling for Code subsection**
  - CodePRM: Execution feedback for process rewards
  - μCODE: Multi-turn generation with verifiers
  - P-GRPO: Mitigating reward hacking
  - Position our peer-review approach

- [ ] **3.3 Quality-Diversity Algorithms subsection**
  - MAP-Elites for diverse elite maintenance
  - Island-based evolution for parallel exploration
  - Our contribution: QD + negative selection

- [ ] **3.4 Negative Selection in EC subsection**
  - Brief coverage of negative selection in artificial immune systems
  - Gap: not applied to LLM-guided program evolution

## Task Group 4: Write Method Section

- [ ] **4.1 System Overview**
  - High-level pipeline description
  - Component interaction diagram
  - Data flow through the system

- [ ] **4.2 Peer-Review Reward Model**
  - Training data: ICLR 2017-2024 reviews
  - Model architecture: Qwen2.5-7B fine-tuning
  - Integration: Proposal filtering at threshold 5.5
  - Evaluation: 72% accuracy on ICLR 2025 holdout

- [ ] **4.3 Toxic Trait Tracking (Core Contribution)**
  - Algorithm pseudocode
  - Key design decision: BEST vs PARENT comparison
  - Dynamic baseline effect visualization
  - O(1) lookup implementation detail

- [ ] **4.4 Failure-Driven Learning**
  - Failure record schema
  - LLM root cause analysis prompt
  - Prompt injection strategy
  - Memory management (last N failures)

- [ ] **4.5 Evolution Infrastructure**
  - MAP-Elites feature map (score × complexity)
  - Island-based evolution (5 islands, migration)
  - Sampling with toxic exclusion
  - Database persistence

## Task Group 5: Write Experiments Section

- [ ] **5.1 Benchmark Problems**
  - Describe AlphaResearchComp (8 problems)
  - Problem characteristics table
  - Evaluation metrics (Excel@best)

- [ ] **5.2 Experimental Setup**
  - Hardware configuration (A100 80GB)
  - LLM configuration (OpenRouter, Llama-3.1-70B)
  - Hyperparameters table
  - Number of runs per condition

- [ ] **5.3 Baselines**
  - OpenEvolve configuration
  - ShinkaEvolve configuration
  - AlphaResearch configuration
  - Fair comparison protocols

- [ ] **5.4 Ablation Study Design**
  - Ablation 1: Toxic tracking on/off
  - Ablation 2: Threshold sensitivity
  - Ablation 3: BEST vs PARENT baseline
  - Ablation 4: Proposal filtering on/off

## Task Group 6: Generate Results

- [ ] **6.1 Run main experiments (if not already done)**
  - Execute Omega Evolve on all 8 benchmarks
  - Collect metrics: best score, iterations, compute time
  - Record per-iteration logs

- [ ] **6.2 Run ablation experiments**
  - Toxic tracking ablation (2 conditions × 8 problems)
  - Threshold sensitivity (5 values × 3 problems)
  - Baseline type ablation (2 conditions × 3 problems)

- [ ] **6.3 Compile baseline results**
  - Run OpenEvolve on same benchmarks (or cite paper)
  - Run ShinkaEvolve (or cite paper)
  - Get AlphaResearch numbers from paper

- [ ] **6.4 Generate figures**
  - Figure 1: System architecture diagram
  - Figure 2: Convergence curves (8 plots)
  - Figure 3: Threshold sensitivity plot
  - Figure 4: Wasted iterations comparison

- [ ] **6.5 Generate tables**
  - Table 1: Main results on AlphaResearchComp
  - Table 2: Ablation - Toxic trait tracking
  - Table 3: Compute efficiency comparison
  - Table 4: Hyperparameters

## Task Group 7: Write Results Section

- [ ] **7.1 Main results analysis**
  - Compare Omega vs baselines on all 8 problems
  - Highlight where we beat human best
  - Discuss failure cases

- [ ] **7.2 Ablation results analysis**
  - Toxic tracking impact quantification
  - Threshold sensitivity findings
  - Baseline comparison type findings

- [ ] **7.3 Compute efficiency analysis**
  - Iterations saved by toxic tracking
  - API cost reduction from proposal filtering
  - Wall-clock time comparison

- [ ] **7.4 Case study**
  - Deep dive on circle packing (n=32)
  - Show evolution trajectory
  - Highlight how toxic tracking helped

## Task Group 8: Write Discussion and Conclusion

- [ ] **8.1 Limitations**
  - Benchmark specificity
  - Threshold sensitivity to domain
  - Reward model bias
  - Compute requirements

- [ ] **8.2 Broader Impact**
  - Positive: Democratizes algorithm discovery
  - Positive: Reduces compute waste
  - Negative: Potential misuse for adversarial code

- [ ] **8.3 Future Work**
  - Adaptive thresholds
  - Similarity-based rejection
  - Cross-benchmark transfer

- [ ] **8.4 Conclusion**
  - Summarize contributions
  - Restate key results
  - Call to action for reproducibility

## Task Group 9: Polish and Finalize

- [ ] **9.1 Write abstract (final version)**
  - 250 words max
  - Problem → Approach → Results → Impact

- [ ] **9.2 Compile references**
  - Verify all citations are accurate
  - Check for missing references
  - Format according to venue style

- [ ] **9.3 Proofread entire paper**
  - Grammar and spelling
  - Consistency in terminology
  - Figure/table references

- [ ] **9.4 Check page limits**
  - Main paper: 8-10 pages
  - Appendix: Unlimited
  - Move overflow to appendix

- [ ] **9.5 Create supplementary materials**
  - Code repository link
  - Trained model weights
  - Benchmark data
  - Reproducibility checklist

## Task Group 10: Appendix

- [ ] **10.1 Implementation Details**
  - Full algorithm pseudocode
  - Configuration file examples
  - API integration details

- [ ] **10.2 Additional Results**
  - Full ablation tables
  - Per-iteration metrics
  - Failure pattern analysis

- [ ] **10.3 Reproducibility**
  - Hardware requirements
  - Software dependencies
  - Random seed information
  - Expected runtime

---

## Priority Order

1. **Task Group 4** (Method) - Core technical content
2. **Task Group 6** (Generate Results) - Data needed for all sections
3. **Task Group 7** (Results) - Key findings
4. **Task Group 2** (Introduction) - Frame the narrative
5. **Task Group 3** (Related Work) - Position in literature
6. **Task Group 5** (Experiments) - Methodology details
7. **Task Group 8** (Discussion/Conclusion) - Synthesis
8. **Task Group 1** (Infrastructure) - Can parallelize
9. **Task Group 9** (Polish) - Final pass
10. **Task Group 10** (Appendix) - Supporting material

---

## Dependencies

- Task Group 6 must complete before Task Group 7
- Task Group 4 should complete before Task Group 2 (need method details for intro)
- Task Group 1 can run in parallel with everything
- Task Group 9 must be last
