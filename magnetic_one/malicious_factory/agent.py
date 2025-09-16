# file: malicious_factory/agent.py

from dataclasses import dataclass, field
from typing import Literal

# This file defines the data structure for a malicious agent's configuration.
# It acts as a standardized data container that is created by the factory
# and consumed by the system runner/wrapper.

DifficultyLevel = Literal[1, 2, 3]
Method = Literal[1, 2, 3]

@dataclass
class MaliciousAgent:
    """A data structure to hold all information about a malicious agent."""
    target_agent_role: str
    target_agent_index: int
    method: Method
    difficulty_level: DifficultyLevel
    prompt: str # This holds the malicious prompt OR the corruption template
    description: str = field(repr=False) # A human-readable description of the strategy