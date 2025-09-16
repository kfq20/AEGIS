# file: malicious_factory/base_strategy.py

from abc import ABC, abstractmethod
from typing import Tuple

# Forward declarations for type hints
class Task: pass
class BaseLLM: pass

class BaseMaliciousStrategy(ABC):
    """
    An abstract base class for a malicious agent generation strategy.
    Each attack method (sabotage, corruption, injection) will be a concrete
    implementation of this class.
    """
    @abstractmethod
    async def generate(
        self,
        task: Task,
        target_agent_role: str,
        target_agent_index: int,
        difficulty: int
    ) -> Tuple[str, str]:
        """
        Generates the malicious prompt and a description of the strategy.

        Args:
            task: The context of the current task.
            target_agent_role: The role of the agent to be corrupted.
            difficulty: The difficulty level (1, 2, 3) for this strategy.

        Returns:
            A tuple containing:
                - prompt (str): The generated malicious prompt.
                - description (str): A human-readable description of the malice.
        """
        pass