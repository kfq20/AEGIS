# file: malicious_factory/enhanced_adapter.py

"""
Enhanced Malicious Injection Adapter

This adapter provides backward compatibility with the existing malicious injection interface
while using the enhanced OWL-style injection logic.
"""

from typing import Dict, Any, Optional
from .enhanced_factory import EnhancedMaliciousFactory, EnhancedMaliciousAgent
from .agent import MaliciousAgent

class EnhancedMaliciousAdapter:
    """
    Adapter that provides backward compatibility with the existing MaliciousAgentFactory interface
    while using the enhanced injection logic.
    """
    
    def __init__(self, llm=None):
        self.enhanced_factory = EnhancedMaliciousFactory(llm=llm)
        self.llm = llm

    async def create_agent(
        self,
        task_query: str,
        target_agent_role: str,
        target_agent_index: int,
        method: int,

    ) -> MaliciousAgent:
        """
        Create a malicious agent using the enhanced factory but return the old format.
        
        Args:
            task_query: The task query
            target_agent_role: Target agent role
            target_agent_index: Target agent index
            method: Injection method (1, 2, 3) - maps to FM types

        
        Returns:
            MaliciousAgent: Compatible with existing interface
        """
        # Map old method numbers to new FM types
        method_mapping = {
            1: "FM-1.1",  # Sabotage -> Task specification deviation
            2: "FM-2.3",  # Corruption -> Deviate from main goal  
            3: "FM-2.4",  # Injection -> Hide important information
        }
        
        injection_type = method_mapping.get(method, "FM-2.3")
        injection_mode = "input"  # Default to input injection for compatibility
        
        # Create enhanced agent
        enhanced_agent = await self.enhanced_factory.create_agent(
            task_query=task_query,
            target_agent_role=target_agent_role,
            target_agent_index=target_agent_index,
            injection_type=injection_type,
            injection_mode=injection_mode,

        )
        
        # Convert to old format
        return MaliciousAgent(
            target_agent_role=enhanced_agent.target_agent_role,
            target_agent_index=enhanced_agent.target_agent_index,
            method=method,  # Keep original method number for compatibility

            prompt=enhanced_agent.description,  # Use description as prompt for compatibility
            description=enhanced_agent.description
        )

    def inject_malicious_behavior(self, original_call_func, malicious_agent: MaliciousAgent, **context):
        """
        Enhanced injection logic with backward compatibility.
        
        Args:
            original_call_func: Original function to call
            malicious_agent: Malicious agent in old format
            **context: Additional context
        
        Returns:
            str: Injected response
        """
        # Map old method to new injection type
        method_mapping = {
            1: "FM-1.1",
            2: "FM-2.3", 
            3: "FM-2.4",
        }
        
        injection_type = method_mapping.get(malicious_agent.method, "FM-2.3")
        injection_mode = "input"  # Default to input injection
        
        # Create enhanced agent for injection
        enhanced_agent = EnhancedMaliciousAgent(
            target_agent_role=malicious_agent.target_agent_role,
            target_agent_index=malicious_agent.target_agent_index,
            injection_type=injection_type,
            injection_mode=injection_mode,

            description=malicious_agent.description,
            agent_context={}  # Will be populated during injection
        )
        
        # Use enhanced injection logic
        return self.enhanced_factory.inject_malicious_behavior(
            original_call_func, enhanced_agent, **context
        )

# Backward compatibility: Export the adapter as MaliciousAgentFactory
MaliciousAgentFactory = EnhancedMaliciousAdapter 