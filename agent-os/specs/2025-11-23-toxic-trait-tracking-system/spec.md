# Specification: Toxic Trait Tracking System

## Goal

Implement negative selection pressure for the evolutionary code optimization system to prevent repeatedly exploring failed solution spaces, reducing wasted compute and improving search efficiency by marking underperforming programs as "toxic" and excluding them from breeding.

## User Stories

- As a researcher running evolution experiments, I want failed approaches to be remembered so the system doesn't waste GPU time on similar bad ideas
- As a system operator, I want to configure the performance threshold that defines "toxic" programs based on my benchmark's characteristics
- As an LLM proposal generator, I want access to failure history so I can avoid suggesting approaches that already failed
- As a data analyst, I want persistent failure records per benchmark so I can analyze patterns in failed evolutionary branches

## Core Requirements

- Configure minimum acceptable performance ratio (default 85% of parent score)
- Mark programs below threshold as "toxic" and exclude from parent/inspiration sampling
- Store failure history in JSON format per benchmark with proposal context and metrics
- Inject failure history into LLM proposal generation prompts
- Maintain compatibility with MAP-Elites, island-based evolution, and archive mechanisms
- Track toxic trait metadata (proposal, parent metrics, child metrics, reason for failure)

## Visual Design

No visual mockups provided - this is a backend system component.

## Reusable Components

### Existing Code to Leverage

**Configuration System** (`evolve_agent/config.py`):
- Dataclass-based config with YAML loading (lines 53-459)
- Environment variable expansion support (lines 15-50)
- Nested configuration structure (DatabaseConfig, EvaluatorConfig, etc.)
- Pattern: Add new `ToxicTraitConfig` dataclass and integrate into main `Config`

**Database Sampling** (`evolve_agent/database.py`):
- `_sample_parent()` (line 778) and `_sample_inspirations()` (line 907) methods
- `_filter_error_programs()` (line 758) pattern for filtering
- Program dataclass with metadata field (line 40-72)
- JSON persistence in `save()` method (line 354)
- Pattern: Add toxic trait filtering similar to error program filtering

**Program Metrics Comparison** (`evolve_agent/database.py`):
- `_is_better()` method (line 636) compares programs using combined_score
- `safe_numeric_average()` utility for metric aggregation
- Pattern: Use same comparison logic to detect underperformance

**Prompt Building** (`evolve_agent/prompt/sampler.py`):
- `build_prompt()` method (line 47) with kwargs support
- Template-based prompt generation with variable substitution
- `_format_evolution_history()` pattern for formatting context
- Pattern: Add failure history formatting similar to evolution history

**Evaluation Flow** (`evolve_agent/controller.py`):
- Child program evaluation in main loop (line 692)
- Parent-child comparison after evaluation
- Database add with metadata (line 757)
- Pattern: Insert toxic trait check after evaluation, before database add

### New Components Required

**FailureTracker Class** (`evolve_agent/failure_tracker.py`):
- Doesn't exist yet - needs to be created
- Why: Centralized management of toxic trait persistence and querying
- Functionality: Load/save failure JSON, track toxic programs, provide query interface

**Toxic Trait Detection Logic**:
- Doesn't exist yet - needs integration in controller.py
- Why: Core business logic to compare child vs parent and mark toxic programs
- Functionality: Calculate performance ratio, mark programs, store failure metadata

**Prompt Template Updates**:
- New template section for "Failed Approaches" doesn't exist
- Why: LLMs need structured failure information in proposal prompts
- Functionality: Format failure history as concise prompt section

## Technical Approach

### Architecture Design

**1. Configuration Layer**

Add `ToxicTraitConfig` dataclass to `evolve_agent/config.py`:
```
@dataclass
class ToxicTraitConfig:
    enabled: bool = True
    threshold: float = 0.85  # 85% of parent score
    comparison_metric: str = "combined_score"  # or "normalized_average"
    failure_history_path: Optional[str] = None  # defaults to db_path/failures/
    max_failures_in_prompt: int = 10  # limit LLM context size
```

Integrate into main Config class alongside other component configs (llm, database, evaluator, etc.)

**2. Data Structures**

Failure record format (stored in JSON):
```json
{
  "program_id": "uuid",
  "parent_id": "parent_uuid",
  "timestamp": 1234567890.0,
  "iteration": 42,
  "proposal": ["proposal text lines"],
  "parent_metrics": {"combined_score": 0.85, ...},
  "child_metrics": {"combined_score": 0.70, ...},
  "performance_ratio": 0.824,
  "threshold": 0.85,
  "reason": "Performance below threshold",
  "comparison_metric": "combined_score"
}
```

File structure: `{db_path}/failures/{benchmark_name}_failures.json`

**3. FailureTracker Component**

New file: `evolve_agent/failure_tracker.py`

Key methods:
- `__init__(config: ToxicTraitConfig, db_path: str)` - Initialize with config and file path
- `add_failure(program: Program, parent: Program, reason: str)` - Record toxic program
- `is_toxic(program_id: str) -> bool` - Check if program is toxic
- `get_failure_history(limit: int) -> List[Dict]` - Get recent failures for prompt
- `save()` - Persist to JSON
- `load()` - Load from JSON

Store toxic program IDs in memory set for fast lookup: `self.toxic_programs: Set[str]`

**4. Integration Points**

**Controller Integration** (in `evolve_agent/controller.py`):

After child evaluation (around line 693), before database add (line 757):
```
# Check toxic trait threshold
if config.toxic_trait.enabled:
    if failure_tracker.should_mark_toxic(child_program, parent):
        failure_tracker.add_failure(child_program, parent, "Below performance threshold")
        logger.warning(f"Marked {child_id} as toxic (below threshold)")
```

**Database Sampling Integration** (in `evolve_agent/database.py`):

Modify `_sample_parent()` and `_sample_inspirations()`:
- Accept optional `toxic_programs: Set[str]` parameter
- Filter out toxic program IDs similar to `_filter_error_programs()` pattern
- Use: `valid_programs = [pid for pid in candidates if pid not in toxic_programs]`

**Prompt Integration** (in `evolve_agent/prompt/sampler.py`):

Add to `build_prompt()` method:
- Accept `failure_history: List[Dict]` in kwargs
- Call `_format_failure_history(failure_history)` method
- Inject formatted failures into user prompt template

Add new method:
```python
def _format_failure_history(self, failures: List[Dict]) -> str:
    """Format failure history for LLM prompt"""
    if not failures:
        return "No previous failures recorded."

    formatted = ["Previously Failed Approaches (avoid these):"]
    for f in failures[:self.config.toxic_trait.max_failures_in_prompt]:
        ratio = f['performance_ratio']
        proposal_summary = f['proposal'][0][:100] if f['proposal'] else "No proposal"
        formatted.append(f"- {proposal_summary} (achieved {ratio:.1%} of parent performance)")

    return "\n".join(formatted)
```

**5. Persistence Strategy**

- Single JSON file per benchmark: `{benchmark_name}_failures.json`
- Append-only writes during evolution for safety
- Periodic compaction to limit file size (keep last 1000 failures)
- Load once at controller initialization
- Save after each toxic trait detection
- File format: JSON array of failure records

**6. Performance Considerations**

- In-memory set lookup: O(1) for toxic program checks
- Lazy loading: Only load failures for active benchmark
- Bounded memory: Limit max failures stored (e.g., 1000 most recent)
- Minimal overhead: Single comparison per child program
- No impact on MAP-Elites grid or archive logic

### Edge Cases and Failure Handling

**Edge Case 1: Parent program is itself toxic**
- Should still allow sampling (toxic programs already filtered in sampling)
- Should not double-penalize descendants

**Edge Case 2: Initial program below threshold**
- Don't mark as toxic (has no parent to compare against)
- Special handling in controller: `if program.parent_id is None: skip toxic check`

**Edge Case 3: Metrics missing or incomparable**
- If child has error metrics, already handled by error filtering (don't mark as toxic)
- If metrics missing, fall back to normalized average comparison
- Log warning if comparison impossible, skip toxic marking

**Edge Case 4: Threshold = 1.0 (100% required)**
- Valid configuration - very strict selection pressure
- Any regression marks program as toxic

**Edge Case 5: Threshold = 0.0 (disabled)**
- Effectively disables toxic trait tracking
- No programs marked as toxic
- Alternative: Use `enabled: false` flag

**Edge Case 6: File corruption or missing failure file**
- Catch JSON decode errors, log warning, continue with empty failure list
- Don't crash evolution run due to failure tracking issues

**Edge Case 7: Island migration with toxic programs**
- Toxic programs already excluded from sampling
- Migration creates copies - copies should also be checked
- Store toxic trait in program metadata for tracking across islands

### Testing Strategy

**Unit Tests** (`tests/test_failure_tracker.py`):
- Test FailureTracker initialization and configuration
- Test adding failures and JSON serialization
- Test toxic program lookup (is_toxic method)
- Test failure history retrieval with limits
- Test file persistence and loading
- Test edge cases (missing metrics, corrupt JSON)

**Integration Tests** (`tests/test_toxic_trait_integration.py`):
- Test controller integration: child evaluation -> toxic check -> database
- Test database sampling filters out toxic programs
- Test prompt injection of failure history
- Test configuration loading from YAML
- Test island migration with toxic programs
- Test MAP-Elites compatibility (toxic programs excluded from grid)

**Performance Tests**:
- Benchmark toxic program lookup with 1000+ entries (should be < 1ms)
- Measure overhead of toxic trait check per iteration (target < 5% slowdown)
- Test memory usage with large failure history (should stay under 10MB)

**Regression Tests**:
- Verify MAP-Elites still converges with toxic trait tracking enabled
- Verify archive still maintains elite programs correctly
- Verify best program tracking unaffected by toxic trait filtering
- Compare evolution convergence with/without toxic trait tracking

### Success Criteria

- Toxic programs correctly excluded from `_sample_parent()` and `_sample_inspirations()`
- Failure history persists across evolution runs (load checkpoint -> failures still present)
- LLM prompts include formatted failure history when generating proposals
- Configuration threshold adjustable via YAML without code changes
- System compatible with existing MAP-Elites grid, island populations, and archive
- No crashes or data loss when failure file is missing or corrupted
- Performance overhead less than 5% per iteration

## Out of Scope

- Automated threshold tuning based on benchmark characteristics (future enhancement)
- Similarity-based failure matching (e.g., "this proposal is similar to a failed one") - requires embedding models
- Failure pattern analysis and clustering (future research feature)
- Toxic trait recovery mechanism (allow retrying failed approaches after N generations)
- Multi-benchmark failure sharing (failures from benchmark A informing benchmark B)
- Visual dashboard for failure analysis (future UI work)
- Integration with external experiment tracking systems (W&B, MLflow)

## Implementation Notes

**File Locations**:
- New: `/Users/overtime/Documents/GitHub/omega-research/evolve_agent/failure_tracker.py`
- Modify: `/Users/overtime/Documents/GitHub/omega-research/evolve_agent/config.py`
- Modify: `/Users/overtime/Documents/GitHub/omega-research/evolve_agent/controller.py`
- Modify: `/Users/overtime/Documents/GitHub/omega-research/evolve_agent/database.py`
- Modify: `/Users/overtime/Documents/GitHub/omega-research/evolve_agent/prompt/sampler.py`
- New tests: `/Users/overtime/Documents/GitHub/omega-research/tests/test_failure_tracker.py`

**Configuration Example** (add to benchmark YAML configs):
```yaml
toxic_trait:
  enabled: true
  threshold: 0.85
  comparison_metric: "combined_score"
  failure_history_path: null  # defaults to db_path/failures/
  max_failures_in_prompt: 10
```

**Dependencies**:
- No new external dependencies required
- Uses existing: json, dataclasses, logging, pathlib, typing
- Reuses existing utilities: safe_numeric_average, format_metrics_safe

**Backward Compatibility**:
- Default config has `enabled: true` but threshold is lenient (0.85)
- Existing checkpoints load correctly (failures/ directory is separate)
- Existing code paths unaffected when `enabled: false`
- No database schema changes (uses program metadata dict)
