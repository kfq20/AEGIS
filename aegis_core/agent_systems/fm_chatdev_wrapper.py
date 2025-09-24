# file: agent_systems/fm_chatdev_wrapper.py

import re
import asyncio
from .base_wrapper import SystemWrapper
from malicious_factory import FMMaliciousFactory, FMMaliciousAgent, FMErrorType, InjectionStrategy
from typing import Any, Dict, Tuple

from methods import get_method_class

class FMChatDevWrapper(SystemWrapper):
    """
    A wrapper for the ChatDev system using the new FM malicious injection system.
    Supports 28 different malicious injection methods across different ChatDev roles.
    """
    def __init__(self, general_config: Dict[str, Any], method_config: Dict[str, Any]):
        from utils.async_llm import create_llm_instance
        exp_config = general_config.get('experiment_config', {})
        llm_config = exp_config.get('llm_config', {})
        if not llm_config:
            raise ValueError("No LLM configuration found for the specified model name.")

        self.llm = create_llm_instance(llm_config)
        
        method_name = exp_config['system_under_test']['name']  # "chatdev"
        dataset_name = exp_config.get('benchmark_name', None)
        MAS_CLASS = get_method_class(method_name, dataset_name)
        self.chatdev_instance = MAS_CLASS(general_config, method_config_name=None)

        self.fm_factory = FMMaliciousFactory(llm=self.llm)

        print(f"FMChatDevWrapper initialized with {MAS_CLASS.__name__}.")

    def run_with_multi_injection(
        self,
        task: Any,
        malicious_agents: list,
        injection_targets: list
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run ChatDev experiment with multiple malicious agent injections.
        """
        if len(malicious_agents) == 1 and len(injection_targets) == 1:
            # Fall back to single injection for backward compatibility
            return self.run_with_injection(task, malicious_agents[0], injection_targets[0])
        
        # Create a mapping from role to malicious agent
        injection_map = {}
        for agent, target in zip(malicious_agents, injection_targets):
            role_name = target['role']
            injection_map[role_name] = {
                'agent': agent,
                'target': target
            }
        
        # Store the original call_llm method
        original_llm_call = self.chatdev_instance.call_llm
        
        # Track conversation history for logging
        conversation_history = []
        
        # --- Define the multi-target FM malicious call_llm method ---
        def fm_multi_malicious_llm_call(*args, **kwargs):
            # Extract arguments - ChatDev uses: call_llm(None, None, messages)
            prompt = args[0] if len(args) > 0 else kwargs.get('prompt', None)
            system_prompt = args[1] if len(args) > 1 else kwargs.get('system_prompt', None)
            messages = args[2] if len(args) > 2 else kwargs.get('messages', [])
            
            # Determine current role from system message or conversation context
            current_role = self._extract_current_role(messages, system_prompt)
            
            print(f"[FM Multi-ChatDev Runner Intercept] Current role: '{current_role}'")
            
            # Check if this role should be injected
            if current_role not in injection_map:
                response = original_llm_call(*args, **kwargs)
                # Log normal conversation
                conversation_history.append({
                    "role": current_role,
                    "messages": messages,
                    "response": response,
                    "injected": False
                })
                return response
            
            # Get the malicious agent and target for this injection
            injection_info = injection_map[current_role]
            malicious_agent = injection_info['agent']
            target = injection_info['target']
            
            print(f"*** FM Multi-Malicious ChatDev Agent Activated on '{current_role}' ***")
            print(f"*** Type: {malicious_agent.fm_error_type.value}, Strategy: {malicious_agent.injection_strategy.value} ***")
            
            # Create mock agent for this role
            mock_agent = MockChatDevAgent(current_role, self.chatdev_instance)
            
            # Perform FM injection
            if malicious_agent.injection_strategy == InjectionStrategy.PROMPT_INJECTION:
                # Modify the input prompt/messages
                modified_messages = self._modify_messages_for_injection(
                    messages, malicious_agent, mock_agent, task
                )
                modified_kwargs = kwargs.copy()
                if len(args) > 2:
                    modified_args = list(args)
                    modified_args[2] = modified_messages
                    response = original_llm_call(*modified_args, **modified_kwargs)
                else:
                    modified_kwargs['messages'] = modified_messages
                    response = original_llm_call(*args, **modified_kwargs)
                
            elif malicious_agent.injection_strategy == InjectionStrategy.RESPONSE_CORRUPTION:
                # Get clean response first, then corrupt it
                clean_response = original_llm_call(*args, **kwargs)
                response = self._corrupt_response(
                    clean_response, malicious_agent, mock_agent, task
                )
            else:
                # Fallback to original response
                response = original_llm_call(*args, **kwargs)
            
            # Log injected conversation
            conversation_history.append({
                "role": current_role,
                "messages": messages,
                "response": response,
                "injected": True,
                "fm_error_type": malicious_agent.fm_error_type.value,
                "injection_strategy": malicious_agent.injection_strategy.value
            })
            
            return response
        
        # --- Monkey patch ---
        self.chatdev_instance.call_llm = fm_multi_malicious_llm_call
        print(f"[FM Multi-ChatDev Runner] Monkey-patch applied to call_llm method.")
        
        # Run the experiment
        sample = {"query": task.query}
        final_output = self.chatdev_instance.inference(sample)
        
        # Restore original method
        self.chatdev_instance.call_llm = original_llm_call
        print("[FM Multi-ChatDev Runner] Original `call_llm` method restored.")
        
        # Build comprehensive log
        log = {
            "final_output": final_output,
            "conversation_history": conversation_history,
            "injected_roles": list(injection_map.keys()),
            "full_history": getattr(self.chatdev_instance, 'history', []),
            "env_dict": getattr(self.chatdev_instance, 'env_dict', {}),
        }
        
        return final_output, log

    def run_with_injection(
        self,
        task: Any,
        malicious_agent: FMMaliciousAgent,
        injection_target: dict
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run ChatDev experiment with single malicious agent injection.
        """
        target_role = injection_target['role']
        
        # Store the original call_llm method
        original_llm_call = self.chatdev_instance.call_llm
        
        # Track conversation history for logging
        conversation_history = []
        
        # --- Define the FM malicious call_llm method ---
        def fm_malicious_llm_call(*args, **kwargs):
            # Extract arguments
            prompt = args[0] if len(args) > 0 else kwargs.get('prompt', None)
            system_prompt = args[1] if len(args) > 1 else kwargs.get('system_prompt', None)
            messages = args[2] if len(args) > 2 else kwargs.get('messages', [])
            
            # Determine current role
            current_role = self._extract_current_role(messages, system_prompt)
            
            print(f"[FM ChatDev Runner Intercept] Current role: '{current_role}', Target: '{target_role}'")
            
            # Check if this is the target role
            if current_role != target_role:
                response = original_llm_call(*args, **kwargs)
                conversation_history.append({
                    "role": current_role,
                    "messages": messages,
                    "response": response,
                    "injected": False
                })
                return response
            
            print(f"*** FM Malicious ChatDev Agent Activated on '{current_role}' ***")
            print(f"*** Type: {malicious_agent.fm_error_type.value}, Strategy: {malicious_agent.injection_strategy.value} ***")
            
            # Create mock agent for this role
            mock_agent = MockChatDevAgent(current_role, self.chatdev_instance)
            
            # Perform FM injection
            if malicious_agent.injection_strategy == InjectionStrategy.PROMPT_INJECTION:
                modified_messages = self._modify_messages_for_injection(
                    messages, malicious_agent, mock_agent, task
                )
                modified_kwargs = kwargs.copy()
                if len(args) > 2:
                    modified_args = list(args)
                    modified_args[2] = modified_messages
                    response = original_llm_call(*modified_args, **modified_kwargs)
                else:
                    modified_kwargs['messages'] = modified_messages
                    response = original_llm_call(*args, **modified_kwargs)
                
            elif malicious_agent.injection_strategy == InjectionStrategy.RESPONSE_CORRUPTION:
                clean_response = original_llm_call(*args, **kwargs)
                response = self._corrupt_response(
                    clean_response, malicious_agent, mock_agent, task
                )
            else:
                response = original_llm_call(*args, **kwargs)
            
            conversation_history.append({
                "role": current_role,
                "messages": messages,
                "response": response,
                "injected": True,
                "fm_error_type": malicious_agent.fm_error_type.value,
                "injection_strategy": malicious_agent.injection_strategy.value
            })
            
            return response
        
        # --- Monkey patch ---
        self.chatdev_instance.call_llm = fm_malicious_llm_call
        print(f"[FM ChatDev Runner] Monkey-patch applied. Target: {target_role}.")
        print(f"[FM ChatDev Runner] Injection: {malicious_agent.fm_error_type.value} via {malicious_agent.injection_strategy.value}")
        
        # Run the experiment
        sample = {"query": task.query}
        final_output = self.chatdev_instance.inference(sample)
        
        # Restore original method
        self.chatdev_instance.call_llm = original_llm_call
        print("[FM ChatDev Runner] Original `call_llm` method restored.")
        
        # Build log
        log = {
            "final_output": final_output,
            "conversation_history": conversation_history,
            "injected_role": target_role,
            "fm_error_type": malicious_agent.fm_error_type.value,
            "injection_strategy": malicious_agent.injection_strategy.value,
            "malicious_action_description": malicious_agent.description,
            "full_history": getattr(self.chatdev_instance, 'history', []),
            "env_dict": getattr(self.chatdev_instance, 'env_dict', {}),
        }
        
        return final_output, log

    def _extract_current_role(self, messages, system_prompt):
        """
        Extract the current role from system prompt or messages.
        ChatDev roles include: Chief Executive Officer, Chief Product Officer, 
        Chief Technology Officer, Programmer, Code Reviewer, Software Test Engineer
        """
        roles = [
            "Chief Executive Officer", "Chief Product Officer", "Chief Technology Officer",
            "Programmer", "Code Reviewer", "Software Test Engineer"
        ]
        
        # Check system prompt first
        if system_prompt:
            for role in roles:
                if role in system_prompt:
                    return role
        
        # Check messages
        if messages:
            for message in messages:
                content = message.get('content', '')
                for role in roles:
                    if role in content:
                        return role
        
        # Check instance history
        if hasattr(self.chatdev_instance, 'history') and self.chatdev_instance.history:
            latest_entry = self.chatdev_instance.history[-1]
            return latest_entry.get('role', 'Unknown')
        
        return "Unknown"

    def _modify_messages_for_injection(self, messages, malicious_agent, mock_agent, task):
        """
        Modify messages for prompt injection.
        """
        if not messages:
            return messages
        
        modified_messages = messages.copy()
        
        # Generate malicious instruction
        malicious_instruction = self.fm_factory.generate_instruction(
            malicious_agent, mock_agent, task
        )
        
        # Inject into the last user message
        if modified_messages and modified_messages[-1].get('role') == 'user':
            original_content = modified_messages[-1]['content']
            modified_messages[-1]['content'] = f"{original_content}\n\n{malicious_instruction}"
        
        return modified_messages

    def _corrupt_response(self, clean_response, malicious_agent, mock_agent, task):
        """
        Corrupt the response for response corruption injection.
        """
        corrupted_response = self.fm_factory.generate_response(
            malicious_agent, mock_agent, task, clean_response
        )
        return corrupted_response


class MockChatDevAgent:
    """
    Mock agent class for ChatDev roles to work with FM factory.
    """
    def __init__(self, role_name, chatdev_instance):
        self.role_name = role_name
        self.role_type = "ChatDev Agent"
        self.agent_id = f"chatdev_{role_name.lower().replace(' ', '_')}"
        self.system_message = self._get_role_description(role_name)
        self.chatdev_instance = chatdev_instance

    def _get_role_description(self, role_name):
        """
        Get description for different ChatDev roles.
        """
        role_descriptions = {
            "Chief Executive Officer": "You are the CEO responsible for high-level decisions and overall project direction.",
            "Chief Product Officer": "You are the CPO responsible for product requirements and specifications.",
            "Chief Technology Officer": "You are the CTO responsible for technical architecture and technology choices.",
            "Programmer": "You are a Programmer responsible for writing and implementing code.",
            "Code Reviewer": "You are a Code Reviewer responsible for reviewing and improving code quality.",
            "Software Test Engineer": "You are a Software Test Engineer responsible for testing and quality assurance."
        }
        return role_descriptions.get(role_name, f"You are a {role_name} in the ChatDev team.")