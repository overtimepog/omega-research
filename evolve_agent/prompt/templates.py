"""
Prompt templates for EvolveAgent
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.
"""

BASE_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert code reviewer.
Your job is to analyze the provided code and evaluate it systematically."""

# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Suggest improvements to the program that will lead to better performance on the specified metrics.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""


DIFF_USER_TEMPLATE_PROPOSAL = """# Previous Proposal: 
{parent_proposal_text}

# Previous Program:
```{language}
{parent_program}
```

# Previous Performance Metrics: 
{metrics}

# Areas Identified for Improvement: 
{improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Proposal
{current_proposal_text}

# Task
Suggest improvements to the program that will lead to better performance on the specified metrics.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""



# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```
"""

# Template for formatting evolution history
EVOLUTION_HISTORY_TEMPLATE = """## Previous Attempts

{previous_attempts}

## Top Performing Programs

{top_programs}
"""

# Template for formatting a previous attempt
PREVIOUS_ATTEMPT_TEMPLATE = """### Attempt {attempt_number}
- Changes: {changes}
- Performance: {performance}
- Outcome: {outcome}
"""

# Template for formatting a top program
TOP_PROGRAM_TEMPLATE = """### Program {program_number} (Score: {score})
```{language}
{program_snippet}
```
Key features: {key_features}
"""

# Template for evaluating a program via an LLM
EVALUATION_TEMPLATE = """Evaluate the following code on a scale of 0.0 to 1.0 for the following metrics:
1. Readability: How easy is the code to read and understand?
2. Maintainability: How easy would the code be to maintain and modify?
3. Efficiency: How efficient is the code in terms of time and space complexity?

For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.

Code to evaluate:
```python
{current_program}
```

Return your evaluation as a JSON object with the following format:
{{
    "readability": [score],
    "maintainability": [score],
    "efficiency": [score],
    "reasoning": "[brief explanation of scores]"
}}
"""

# Template for bug fixing
BUG_FIX_TEMPLATE = """# Bug Fix Task

The following program encountered an error during evaluation and needs to be fixed.

## Research Proposal Context
{proposal_text}

## Error Information
- **Error Type**: {error_type}
- **Error Message**: {error_message}

## Full Traceback
```
{traceback}
```

## Buggy Program
```{language}
{buggy_code}
```

## Task
Analyze the error and generate SEARCH/REPLACE diffs to fix the bug. Focus on:
1. Understanding the root cause from the traceback
2. Making minimal changes to fix the specific error
3. Preserving the research idea and overall program structure
4. Ensuring tensor/array shapes are compatible
5. Handling edge cases that may trigger the error

You MUST use the exact SEARCH/REPLACE diff format:

<<<<<<< SEARCH
# Exact code to find (must match exactly, including indentation)
=======
# Fixed replacement code
>>>>>>> REPLACE

Example fix for tensor size mismatch:
<<<<<<< SEARCH
features = torch.stack([log_freq, recency, uncertainty], dim=-1)
=======
# Ensure all tensors have the same shape before stacking
# Reshape to match the largest sequence length
max_len = max(log_freq.size(0), recency.size(0), uncertainty.size(0))
log_freq = F.pad(log_freq, (0, 0, 0, max_len - log_freq.size(0)))
recency = F.pad(recency, (0, 0, 0, max_len - recency.size(0)))
uncertainty = F.pad(uncertainty, (0, 0, 0, max_len - uncertainty.size(0)))
features = torch.stack([log_freq, recency, uncertainty], dim=-1)
>>>>>>> REPLACE

IMPORTANT:
- Each SEARCH section must exactly match code in the buggy program
- Focus only on fixing the error, not on making other improvements
- Test your fix mentally against the error message and traceback
- Provide clear comments explaining the fix
"""


# System message for changes documentation
CHANGES_DOCUMENTATION_SYSTEM = """You are an expert code reviewer and technical writer specializing in analyzing code evolution.
Your task is to analyze the changes between two versions of a program and create comprehensive, insightful documentation
explaining what changed, why, and how it improved performance.

Focus on:
1. Line-by-line analysis of actual code changes
2. Connecting changes to the research proposal that motivated them
3. Explaining the technical reasoning behind improvements
4. Linking code changes to specific metric improvements
5. Providing clear, educational explanations suitable for developers

Be precise, technical, and insightful. Avoid generic statements - every explanation should be specific to the actual code changes."""


# User message template for changes documentation
CHANGES_DOCUMENTATION_USER = """# Task
Analyze the code evolution between the parent program and the improved child program, then generate comprehensive documentation
explaining what changed, why, and how it improved the performance metrics.

# Parent Program (Before)
```{language}
{parent_code}
```

# Child Program (After - Improved)
```{language}
{child_code}
```

# Unified Diff
```diff
{unified_diff}
```

# Research Proposal That Motivated Changes
{proposal}

# Performance Metrics Comparison

## Parent Metrics (Before)
{parent_metrics}

## Child Metrics (After)
{child_metrics}

# Your Task

Generate a detailed changes documentation in JSON format with the following structure:

{{
    "summary": "A comprehensive 2-3 sentence summary of what changed and the overall impact",
    "changes": [
        {{
            "title": "Brief descriptive title for this change",
            "location": "Lines X-Y",
            "old_code": "The exact old code that was changed",
            "new_code": "The exact new code that replaced it",
            "reason": "Detailed explanation of WHY this change was made, connecting to the research proposal",
            "impact": "Specific explanation of HOW this change improved the metrics, with technical details"
        }}
    ],
    "overall_impact": "A comprehensive analysis of how all changes work together to achieve the metric improvements. Connect specific code changes to specific metric changes. Be technical and precise."
}}

IMPORTANT GUIDELINES:
1. Analyze the ACTUAL diff - don't hallucinate changes that didn't happen
2. For each change, extract the EXACT old and new code from the diff
3. Explain the technical reasoning - why would this specific change improve performance?
4. Connect changes to metric improvements - be specific about which changes affected which metrics
5. If multiple changes work together, explain their synergistic effects
6. Use technical terminology appropriate for developers
7. Focus on the "why" and "how", not just the "what"
8. Return ONLY valid JSON - no markdown formatting, no code blocks, just the JSON object

Generate the documentation now:"""


# Default templates dictionary
DEFAULT_TEMPLATES = {
    "system_message": BASE_SYSTEM_TEMPLATE,
    "evaluator_system_message": BASE_EVALUATOR_SYSTEM_TEMPLATE,
    # "diff_user": DIFF_USER_TEMPLATE,
    "full_rewrite_user": FULL_REWRITE_USER_TEMPLATE,
    "evolution_history": EVOLUTION_HISTORY_TEMPLATE,
    "previous_attempt": PREVIOUS_ATTEMPT_TEMPLATE,
    "top_program": TOP_PROGRAM_TEMPLATE,
    "evaluation": EVALUATION_TEMPLATE,
    "diff_user": DIFF_USER_TEMPLATE_PROPOSAL,
    "bug_fix": BUG_FIX_TEMPLATE,
    "changes_doc_system": CHANGES_DOCUMENTATION_SYSTEM,
    "changes_doc_user": CHANGES_DOCUMENTATION_USER,
}


class TemplateManager:
    """Manages templates for prompt generation"""

    def __init__(self, template_dir: Optional[str] = None):
        self.templates = DEFAULT_TEMPLATES.copy()

        # Load templates from directory if provided
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)

    def _load_templates_from_dir(self, template_dir: str) -> None:
        """Load templates from a directory"""
        for file_path in Path(template_dir).glob("*.txt"):
            template_name = file_path.stem
            with open(file_path, "r") as f:
                self.templates[template_name] = f.read()

    def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template
