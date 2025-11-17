# Changes Documentation - Usage Guide

## Quick Start

The changes documentation feature is **enabled by default**. When a new best solution is found during evolution, the system automatically generates a `best_solution_changes.md` file explaining:
- What code changed (line-by-line)
- Why changes were made (based on the proposal)
- How changes improved metrics (performance analysis)

## Location

When a new best solution is saved, you'll find:
```
output_dir/best_solution/
├── best_solution.py              # The code
├── best_solution_proposal.txt    # The proposal
├── best_solution_info.json       # Metadata
└── best_solution_changes.md      # ← NEW! Automated documentation
```

## Configuration

### Enable/Disable

In your YAML config file:
```yaml
# Disable changes documentation
generate_changes_doc: false
```

Or in code:
```python
from evolve_agent.config import Config

config = Config()
config.generate_changes_doc = False  # Disable
```

### Adjust Retry Attempts

If LLM generation fails, the system retries with exponential backoff. Adjust retries:

```yaml
# Increase retry attempts for more reliable generation
changes_doc_max_retries: 5  # default: 3
```

## Example Output

Here's what the generated documentation looks like:

```markdown
# Code Evolution Changes

## Summary

Replaced naive recursive fibonacci with memoized version, reducing time
complexity from O(2^n) to O(n). Added internal helper function with dictionary
to cache computed values.

## Metrics Improvement

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| execution_time | 2.500000 | 0.002000 | -99.9% |
| memory_usage | 100.000000 | 120.000000 | +20.0% |

## Detailed Changes

### Add memoization for fibonacci calculation

**Location:** Lines 2-10

**What changed:**
```diff
- if n <= 1:
-     return n
- return fibonacci(n-1) + fibonacci(n-2)
+ # Use memoization for better performance
+ memo = {}
+
+ def fib_helper(x):
+     if x in memo:
+         return memo[x]
+     if x <= 1:
+         return x
+     memo[x] = fib_helper(x-1) + fib_helper(x-2)
+     return memo[x]
+
+ return fib_helper(n)
```

**Why:** The original implementation used naive recursion which led to
exponential time complexity. By adding memoization, we cache previously
computed results and avoid redundant calculations.

**Impact:** This change dramatically reduces the time complexity from
O(2^n) to O(n), making the function viable for larger input values.
For n=30, this represents a speedup of over 1000x.

## Overall Impact

The memoization optimization transforms this function from impractical
for large inputs to highly efficient. The slight increase in memory
usage is a worthwhile trade-off for the massive speedup.
```

## How It Works

1. **Evolution runs** and finds a new best solution
2. **Parent program** is retrieved from the database
3. **Unified diff** is computed between parent and child code
4. **LLM is called** with:
   - Parent code (before)
   - Child code (after - improved)
   - Unified diff
   - Research proposal that motivated changes
   - Metrics before and after
5. **LLM analyzes** and generates structured JSON
6. **System converts** JSON to markdown
7. **File is saved** as `best_solution_changes.md`

## Troubleshooting

### No changes.md generated?

**Check 1**: Is the feature enabled?
```python
# Print config
from evolve_agent.config import load_config
config = load_config("path/to/config.yaml")
print(f"Enabled: {config.generate_changes_doc}")
```

**Check 2**: Does the program have a parent?
- The first solution has no parent, so no changes.md is generated
- Only solutions with `parent_id` get documentation

**Check 3**: Check logs
```
# Look for these log messages:
INFO - Generating changes documentation (child: xxx, parent: yyy)
INFO - Successfully generated changes documentation (N changes, M metrics)
INFO - Saved changes documentation to path/to/best_solution_changes.md

# Or warnings:
WARNING - No parent program available for changes documentation
ERROR - Failed to generate changes documentation after 3 attempts
```

### LLM generation failing?

**Solution 1**: Increase retry attempts
```yaml
changes_doc_max_retries: 5  # Up from default 3
```

**Solution 2**: Check LLM configuration
- Ensure API key is valid
- Ensure model supports JSON output
- Check timeout settings

**Solution 3**: Check fallback documentation
- Even if LLM fails, a basic diff-based documentation is generated
- Look for: "The code was modified to improve performance metrics."
- This indicates fallback mode was used

### Documentation quality issues?

**Issue**: Documentation is too generic or missing details

**Solutions**:
1. **Improve proposals**: Better proposals → better documentation
   - Be specific about what optimization is being tried
   - Explain expected impact

2. **Use better LLM model**:
   ```yaml
   llm:
     models:
       - name: gpt-4  # More capable model
   ```

3. **Add more context to metrics**:
   - Include multiple performance metrics
   - Track before/after values carefully

## Advanced Usage

### Programmatic Access

Access the change documentation programmatically:

```python
from evolve_agent.controller import EvolveAgent
from evolve_agent.database import Program

# During evolution
async def on_new_best(program: Program, parent: Program):
    # Generate documentation manually
    changes_md = await controller._generate_changes_documentation(
        child_program=program,
        parent_program=parent
    )

    # Do something with it
    print(changes_md)

    # Or parse it
    # Note: It's markdown, so you can use a markdown parser
```

### Custom Templates

You can customize the LLM prompts by modifying the templates:

```python
from evolve_agent.prompt.templates import TemplateManager

template_manager = TemplateManager()

# Add custom system message
custom_system = """You are an expert code reviewer specializing in Python.
Focus on performance optimizations and algorithmic improvements."""

template_manager.add_template("changes_doc_system", custom_system)
```

### Integration with CI/CD

Automatically commit changes documentation:

```bash
# After evolution run
cd output_dir/best_solution/

# Check if changes.md exists
if [ -f "best_solution_changes.md" ]; then
    # Add to git
    git add best_solution_changes.md
    git commit -m "docs: Update best solution changes documentation"

    # Optionally create PR
    gh pr create --title "New best solution" --body-file best_solution_changes.md
fi
```

## Best Practices

1. **Review the documentation** after each evolution run
   - Verify LLM explanations make sense
   - Check if metric improvements match expectations

2. **Use meaningful proposals**
   - The better your research proposals, the better the documentation
   - Include specific hypotheses about performance impact

3. **Track multiple metrics**
   - More metrics = better impact analysis
   - Include both performance and quality metrics

4. **Keep diffs manageable**
   - Very large diffs (>500 lines) may reduce documentation quality
   - Consider breaking large changes into multiple iterations

5. **Archive documentation**
   - Save `best_solution_changes.md` at each checkpoint
   - Track evolution of optimizations over time

## FAQ

**Q: Does this slow down evolution?**
A: Minimally. The LLM call happens asynchronously and only when a new best solution is found (rare). Typical overhead: 1-3 seconds per best solution.

**Q: Can I use this without an LLM?**
A: Yes, set `generate_changes_doc: false` to disable. The basic code/proposal/metadata files are still saved.

**Q: What happens if the LLM fails?**
A: The system falls back to a basic documentation with the raw diff and metrics table. Evolution continues normally.

**Q: Can I regenerate documentation later?**
A: Yes, you can manually call `_generate_changes_documentation()` with any program pair from the database.

**Q: Does this work with non-Python code?**
A: Yes! The system detects the language and adjusts prompts accordingly. Tested with Python, JavaScript, Java, C++, and Rust.

**Q: Can I export to other formats?**
A: The output is markdown, which can be easily converted to HTML, PDF, or other formats using tools like Pandoc.

---

**Need help?** Check the logs at `output_dir/logs/evolve_agent_*.log` for detailed debugging information.
