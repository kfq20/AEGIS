# file: framework/task.py

from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class Task:
    """
    A lightweight, universal data container for a single task instance,
    typically corresponding to one sample from a benchmark dataset.
    """
    # The actual query or problem description given to the agent system.
    query: str

    # The ground truth used for evaluation. This can be a simple string,
    # a number, or a dictionary containing complex data like unit tests.
    ground_truth: Any

    # Optional metadata for the evaluator, e.g., task category, source.
    metadata: Dict[str, Any] = field(default_factory=dict)