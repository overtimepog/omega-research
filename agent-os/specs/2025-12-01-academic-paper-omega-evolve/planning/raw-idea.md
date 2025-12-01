# Raw Idea: Academic Paper on Omega Research (AlphaResearch Extensions)

## Core Concept

Write a rigorous academic paper documenting the novel contributions of the Omega Research codebase, which extends AlphaResearch with three key innovations:

1. **Toxic Trait Tracking System** - Negative selection pressure with dynamic baseline comparison
2. **Peer-Review Reward Decoupling** - Independent proposal scoring using trained reward models
3. **Automated Code Evolution Documentation** - LLM-generated explanations of code changes

## Target Venue

- **Primary**: NeurIPS 2026 / ICML 2026 (main conference)
- **Alternative**: GECCO 2026 LLMfwEC Workshop, ICLR 2026
- **Format**: 8-10 page conference paper + appendix

## Key Claims to Support

1. Toxic trait tracking reduces wasted compute by filtering failed exploration paths
2. Dynamic baseline comparison (vs. BEST, not parent) creates rising selection pressure
3. Peer-review reward decoupling enables early filtering of low-quality proposals
4. Failure-driven learning prevents repetition of known failure patterns
5. System achieves competitive results on AlphaResearchComp benchmarks

## Distinguishing Factors vs. Prior Work

| System | Our Advantage |
|--------|---------------|
| AlphaEvolve | Open-source, peer-review rewards, toxic trait tracking |
| OpenEvolve | Toxic trait system, failure analysis, automated documentation |
| ShinkaEvolve | Dynamic baseline (rising threshold), LLM failure analysis |
| AlphaResearch | Toxic trait tracking (novel), failure history in prompts |

## Paper Structure Vision

1. Abstract
2. Introduction (motivation, contributions)
3. Related Work (FunSearch → AlphaEvolve → OpenEvolve → ShinkaEvolve → AlphaResearch)
4. Method
   - 4.1 System Overview
   - 4.2 Peer-Review Reward Model
   - 4.3 Toxic Trait Tracking (novel contribution)
   - 4.4 Failure-Driven Learning
   - 4.5 MAP-Elites + Island Evolution
5. Experiments
   - 5.1 Benchmark Problems (AlphaResearchComp)
   - 5.2 Ablation Studies (toxic trait on/off, threshold sensitivity)
   - 5.3 Compute Efficiency Analysis
   - 5.4 Case Studies (circle packing, MSTD)
6. Results & Analysis
7. Discussion & Limitations
8. Conclusion
9. References
10. Appendix (implementation details, additional results)

## Required Experiments (if not already run)

- [ ] Baseline comparison: OpenEvolve vs. Omega on same benchmarks
- [ ] Ablation: toxic trait enabled vs. disabled
- [ ] Ablation: threshold sensitivity (0.7, 0.8, 0.85, 0.9, 0.95)
- [ ] Compute efficiency: iterations to convergence with/without toxic tracking
- [ ] Failure pattern analysis: what types of proposals fail most often?

## Visual Assets Needed

- System architecture diagram
- Evolution loop flowchart
- Toxic trait decision tree
- Results comparison tables
- Convergence curves
- Failure pattern visualization
