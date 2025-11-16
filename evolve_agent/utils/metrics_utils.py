"""
Safe calculation utilities for metrics containing mixed types
"""

import logging
import numpy as np
from typing import Any, Dict, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

# Metric configuration: direction and typical ranges for normalization
METRIC_CONFIG = {
    'perplexity': {
        'direction': 'minimize',
        'range': (1.0, 1000.0),  # Typical range for perplexity
        'use_log': True,  # Use log transformation for exponential metrics
    },
    'accuracy': {
        'direction': 'maximize',
        'range': (0.0, 1.0),
        'use_log': False,
    },
    'train_loss': {
        'direction': 'minimize',
        'range': (0.0, 10.0),
        'use_log': False,
    },
    'val_loss': {
        'direction': 'minimize',
        'range': (0.0, 10.0),
        'use_log': False,
    },
    'loss': {
        'direction': 'minimize',
        'range': (0.0, 10.0),
        'use_log': False,
    },
    'score': {
        'direction': 'maximize',
        'range': (0.0, 1.0),
        'use_log': False,
    },
    'combined_score': {
        'direction': 'maximize',
        'range': (0.0, 1.0),
        'use_log': False,
    },
}


def normalize_metric(
    value: float,
    metric_name: str,
    metric_range: Optional[Tuple[float, float]] = None,
    direction: Optional[Literal['minimize', 'maximize']] = None,
    use_log: Optional[bool] = None,
) -> float:
    """
    Normalize a single metric value to [0, 1] range with proper handling of
    direction (minimize vs maximize) and scale (bounded vs unbounded).

    Based on 2025 best practices for evolutionary optimization:
    - Bounded metrics (accuracy): Direct min-max scaling
    - Unbounded metrics (perplexity): Log transformation + min-max scaling
    - All metrics converted to "higher is better" for consistency

    Args:
        value: The metric value to normalize
        metric_name: Name of the metric (for config lookup)
        metric_range: Optional override for (min, max) range. If None, uses METRIC_CONFIG
        direction: Optional override for 'minimize' or 'maximize'. If None, uses METRIC_CONFIG
        use_log: Optional override for log transformation. If None, uses METRIC_CONFIG

    Returns:
        Normalized value in [0, 1] range where higher is always better
    """
    # Get config or use defaults
    config = METRIC_CONFIG.get(metric_name, {})
    metric_range = metric_range or config.get('range', (0.0, 1.0))
    direction = direction or config.get('direction', 'maximize')
    use_log = use_log if use_log is not None else config.get('use_log', False)

    min_val, max_val = metric_range

    # Handle invalid values
    if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
        logger.warning(f"Invalid value for {metric_name}: {value}, returning 0.0")
        return 0.0

    # Apply log transformation for exponential metrics (like perplexity)
    if use_log and value > 0:
        value = np.log(value)
        min_val = np.log(max(min_val, 1e-10))
        max_val = np.log(max(max_val, 1e-10))

    # Clip to range to handle outliers
    value = np.clip(value, min_val, max_val)

    # Min-max normalization to [0, 1]
    if max_val - min_val > 1e-10:
        normalized = (value - min_val) / (max_val - min_val)
    else:
        normalized = 0.5  # If range is degenerate, use middle value

    # Invert for minimize metrics (so higher normalized value = better)
    if direction == 'minimize':
        normalized = 1.0 - normalized

    return float(np.clip(normalized, 0.0, 1.0))


def normalize_metrics_dict(
    metrics: Dict[str, Any],
    exclude_keys: Optional[set] = None,
) -> Dict[str, float]:
    """
    Normalize all numeric metrics in a dictionary to [0, 1] range.

    All metrics are converted to "higher is better" for consistent comparisons.

    Args:
        metrics: Dictionary of metric names to values
        exclude_keys: Optional set of metric names to exclude from normalization

    Returns:
        Dictionary of normalized metric values (only numeric metrics included)
    """
    exclude_keys = exclude_keys or set()
    normalized = {}

    for name, value in metrics.items():
        if name in exclude_keys:
            continue

        if isinstance(value, (int, float)):
            try:
                normalized[name] = normalize_metric(value, name)
            except Exception as e:
                logger.warning(f"Failed to normalize {name}={value}: {e}")
                continue

    return normalized


def safe_numeric_average(metrics: Dict[str, Any], auto_normalize: bool = False) -> float:
    """
    Calculate the average of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings.

    WARNING: Without normalization, this function can produce misleading results when
    metrics have different scales or optimization directions. Always prefer using
    'combined_score' for comparisons, or set auto_normalize=True.

    Args:
        metrics: Dictionary of metric names to values
        auto_normalize: If True, automatically normalize metrics to [0,1] range with
                       proper direction handling before averaging (recommended)

    Returns:
        Average of numeric values, or 0.0 if no numeric values found
    """
    if not metrics:
        return 0.0

    # If auto_normalize is enabled, use normalized metrics
    if auto_normalize:
        normalized = normalize_metrics_dict(metrics)
        if not normalized:
            return 0.0
        return sum(normalized.values()) / len(normalized)

    # Legacy behavior: raw averaging (can produce misleading results)
    numeric_values = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            try:
                # Convert to float and check if it's a valid number
                float_val = float(value)
                if not (float_val != float_val):  # Check for NaN (NaN != NaN is True)
                    numeric_values.append(float_val)
            except (ValueError, TypeError, OverflowError):
                # Skip invalid numeric values
                continue

    if not numeric_values:
        return 0.0

    return sum(numeric_values) / len(numeric_values)


def safe_numeric_sum(metrics: Dict[str, Any]) -> float:
    """
    Calculate the sum of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Sum of numeric values, or 0.0 if no numeric values found
    """
    if not metrics:
        return 0.0

    numeric_sum = 0.0
    for value in metrics.values():
        if isinstance(value, (int, float)):
            try:
                # Convert to float and check if it's a valid number
                float_val = float(value)
                if not (float_val != float_val):  # Check for NaN (NaN != NaN is True)
                    numeric_sum += float_val
            except (ValueError, TypeError, OverflowError):
                # Skip invalid numeric values
                continue

    return numeric_sum
