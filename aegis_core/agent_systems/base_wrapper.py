# file: agent_systems/base_wrapper.py

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict

# To prevent circular imports while allowing type hinting, we can
# use string-based hints or a simple forward declaration.
# This makes the code cleaner and more maintainable.
class Task:
    pass

class MaliciousAgent:
    pass


class SystemWrapper(ABC):
    """
    An abstract base class for all multi-agent system wrappers.

    This class defines a standard interface for running experiments on different
    agent frameworks. Each specific framework (e.g., AFlow, Magentic-One) should
    have its own concrete implementation inheriting from this class.

    The main goal is to abstract away the specific details of how each
    framework is initialized and executed, allowing the main experiment
    runner to treat them uniformly.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        The constructor should handle the specific setup for the wrapped
        agent system, such as loading configurations or initializing clients.
        
        This is an abstract method because each system will have a unique
        initialization process.
        """
        pass

    @abstractmethod
    def run_with_injection(
        self,
        task: Task,
        malicious_agent: MaliciousAgent,
        injection_target: dict
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        The core method to run an experiment with a malicious agent injected.

        This method must be implemented by all subclasses. It encapsulates the
        entire logic of executing a task within the target agent system while
        ensuring the malicious agent's behavior is correctly inserted.

        Args:
            task (Task): The task object containing the problem description and
                         ground truth criteria.
            malicious_agent (MaliciousAgent): The malicious agent object containing
                                              its prompt and role.
            injection_target (dict): The target specification for injection.

        Returns:
            A tuple containing:
                - final_output (Any): The final result produced by the agent system.
                - conversation_log (Dict[str, Any]): A dictionary or string
                  containing the log of the interaction for later evaluation.
        """
        pass
    
    def run_with_multi_injection(
        self,
        task: Task,
        malicious_agents: list,
        injection_targets: list
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run an experiment with multiple malicious agents injected.
        
        Default implementation falls back to single injection if only one target.
        Subclasses should override this method for better multi-agent support.

        Args:
            task (Task): The task object containing the problem description and
                         ground truth criteria.
            malicious_agents (list): List of malicious agent objects.
            injection_targets (list): List of target specifications for injections.

        Returns:
            A tuple containing:
                - final_output (Any): The final result produced by the agent system.
                - conversation_log (Dict[str, Any]): A dictionary or string
                  containing the log of the interaction for later evaluation.
        """
        if len(malicious_agents) == 1 and len(injection_targets) == 1:
            # Fall back to single injection for backward compatibility
            return self.run_with_injection(task, malicious_agents[0], injection_targets[0])
        else:
            # Default multi-injection implementation - subclasses should override
            raise NotImplementedError("Multi-agent injection not implemented for this wrapper")