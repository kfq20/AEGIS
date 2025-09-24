# file: agent_systems/fm_dylan_wrapper.py

import re
import asyncio
from .base_wrapper import SystemWrapper
from malicious_factory import FMMaliciousFactory, FMMaliciousAgent, FMErrorType, InjectionStrategy
from typing import Any, Dict, Tuple

from methods import get_method_class

class FMDyLANWrapper(SystemWrapper):
    """
    A wrapper for the DyLAN system using the new FM malicious injection system.
    Supports 28 different malicious injection methods.
    """
    def __init__(self, general_config: Dict[str, Any], method_config: Dict[str, Any]):
        from utils.async_llm import create_llm_instance
        exp_config = general_config.get('experiment_config', {})
        llm_config = exp_config.get('llm_config', {})
        if not llm_config:
            raise ValueError("No LLM configuration found for the specified model name.")

        self.llm = create_llm_instance(llm_config)
        method_name = exp_config['system_under_test']['name']  # "dylan"
        dataset_name = exp_config.get('benchmark_name', None)
        MAS_CLASS = get_method_class(method_name, dataset_name)
        self.dylan_instance = MAS_CLASS(general_config, method_config_name=None)

        self.fm_factory = FMMaliciousFactory(llm=self.llm)

        print(f"FMDyLANWrapper initialized with {MAS_CLASS.__name__}.")

    def run_with_multi_injection(
        self,
        task: Any,
        malicious_agents: list,
        injection_targets: list
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run DyLAN experiment with multiple malicious agent injections.
        """
        if len(malicious_agents) == 1 and len(injection_targets) == 1:
            # Fall back to single injection for backward compatibility
            return self.run_with_injection(task, malicious_agents[0], injection_targets[0])
        
        # Create a mapping from (role, role_index) to malicious agent
        injection_map = {}
        for agent, target in zip(malicious_agents, injection_targets):
            key = (target['role'], target.get('role_index', 0))
            injection_map[key] = {
                'agent': agent,
                'target': target
            }
        
        # Store the original call_llm method
        original_llm_call = self.dylan_instance.call_llm
        
        # --- Define the multi-target FM malicious call_llm method ---
        def fm_multi_malicious_llm_call(*args, **kwargs):
            messages = args[2] if len(args) > 2 else kwargs.get('messages', [])
            
            # Get current node information
            node_idx = getattr(self.dylan_instance, 'current_node_idx', None)
            round_id = getattr(self.dylan_instance, 'current_round_id', None)
            num_agents = getattr(self.dylan_instance, 'num_agents', None)
            current_role = None
            current_role_index = None
            
            if node_idx is not None and hasattr(self.dylan_instance, 'nodes'):
                current_role = self.dylan_instance.nodes[node_idx].get('role', None)
                current_role_index = node_idx % num_agents if num_agents else None
            
            print(f"[FM Multi-DyLAN Runner Intercept] Node {node_idx}: role='{current_role}', index={current_role_index}, round={round_id}")
            
            # Check for aggregator injection (Ranker role)
            # Aggregator injection happens during ranking phase (round 2+)
            is_aggregator_call = False
            if round_id is not None and round_id >= 2:
                # Check if this is a ranking call (listwise_ranker)
                messages = args[2] if len(args) > 2 else kwargs.get('messages', [])
                if messages and any("rank" in msg.get('content', '').lower() for msg in messages):
                    is_aggregator_call = True
                    current_role = "Ranker"
                    current_role_index = num_agents  # Use num_agents as aggregator index
            
            # Check if this agent should be injected
            injection_key = (current_role, current_role_index)
            if injection_key not in injection_map and not is_aggregator_call:
                return original_llm_call(*args, **kwargs)
            
            # Get the malicious agent and target for this injection
            if is_aggregator_call:
                # For aggregator injection, find the Ranker target
                aggregator_key = ("Ranker", num_agents)
                if aggregator_key in injection_map:
                    injection_info = injection_map[aggregator_key]
                    malicious_agent = injection_info['agent']
                    target = injection_info['target']
                    print(f"*** FM Multi-Malicious Aggregator Activated on 'Ranker' (round {round_id}) ***")
                    print(f"    Error Type: {malicious_agent.fm_error_type.value}, Strategy: {malicious_agent.injection_strategy.value}")
                else:
                    return original_llm_call(*args, **kwargs)
            else:
                injection_info = injection_map[injection_key]
                malicious_agent = injection_info['agent']
                target = injection_info['target']
                print(f"*** FM Multi-Malicious Agent Activated on '{current_role}' (index {current_role_index}, node {node_idx}, round {round_id}) ***")
                print(f"    Error Type: {malicious_agent.fm_error_type.value}, Strategy: {malicious_agent.injection_strategy.value}")
            
            # Extract agent context
            class MockAgent:
                def __init__(self, role, role_description, dylan_instance, is_aggregator=False):
                    self.role_name = role
                    self.role_type = "DyLAN Agent" if not is_aggregator else "DyLAN Aggregator"
                    self.agent_id = f"dylan_{current_role_index}" if not is_aggregator else f"dylan_ranker"
                    self.system_message = role_description
                    self.tool_dict = {}
                    self.model_type = "LLM"
                    self.chat_history = []
                    
                    self.dylan_instance = dylan_instance
                    self.node_id = node_idx
                    self.round = round_id
            
            # Get role description
            if is_aggregator_call:
                role_description = "You are a DyLAN Ranker agent responsible for ranking and selecting the best responses from other agents."
                mock_agent = MockAgent("Ranker", role_description, self.dylan_instance, is_aggregator=True)
            else:
                role_map = self.dylan_instance._get_role_map()
                role_description = role_map.get(current_role, "You are a DyLAN agent.")
                mock_agent = MockAgent(current_role, role_description, self.dylan_instance)
            
            agent_context = self.fm_factory.extract_agent_context(
                mock_agent,
                {"name": current_role, "description": f"DyLAN {current_role} agent"}
            )
            
            # Update malicious_agent's context
            malicious_agent.agent_context = agent_context
            
            # Apply injection based on strategy
            task_input = messages[-1]['content'] if messages else ""
            
            if malicious_agent.injection_strategy == InjectionStrategy.PROMPT_INJECTION:
                # For prompt injection, modify the prompt first
                modified_prompt = self.fm_factory.inject_prompt(task_input, malicious_agent.fm_error_type, malicious_agent.agent_context)
                
                # Create modified messages
                modified_messages = messages.copy()
                if modified_messages and len(modified_messages) > 0:
                    modified_messages[-1] = modified_messages[-1].copy()
                    modified_messages[-1]['content'] = modified_prompt
                
                # Update kwargs with modified messages
                modified_kwargs = kwargs.copy()
                modified_kwargs['messages'] = modified_messages
                
                # Call original function with modified kwargs
                response = original_llm_call(*args, **modified_kwargs)
                
            elif malicious_agent.injection_strategy == InjectionStrategy.RESPONSE_CORRUPTION:
                # For response corruption, call original function first, then corrupt the response
                clean_response = original_llm_call(*args, **kwargs)
                response = self.fm_factory.corrupt_response(clean_response, malicious_agent.fm_error_type, malicious_agent.agent_context)
            else:
                response = original_llm_call(*args, **kwargs)
            
            return response

        # Apply the monkey patch
        self.dylan_instance.call_llm = fm_multi_malicious_llm_call
        print(f"[FM Multi-DyLAN Runner] Monkey-patch applied for {len(malicious_agents)} targets.")
        
        # Execute the patched workflow
        sample = {"query": task.query}
        final_output = self.dylan_instance.inference(sample)

        # Cleanup
        self.dylan_instance.call_llm = original_llm_call
        print("[FM Multi-DyLAN Runner] Original `call_llm` method restored.")
        
        # Collect history from DyLAN's network structure
        full_history = []
        if hasattr(self.dylan_instance, 'nodes'):
            for i, node in enumerate(self.dylan_instance.nodes):
                if node.get('active', False) and node.get('reply'):
                    role = node.get('role', f'agent_{i}')
                    role_index = i % self.dylan_instance.num_agents
                    full_history.append({
                        "role": role,
                        "role_index": role_index,
                        "content": node.get('reply', ''),
                        "node_id": i,
                        "round": i // self.dylan_instance.num_agents
                    })
        
        # Create comprehensive log with multi-agent information
        log = {
            "final_output": final_output,
            "full_history": full_history,
            "multi_injection_info": [
                {
                    "injected_role": target['role'],
                    "injected_role_index": target.get('role_index', 0),
                    "fm_error_type": agent.fm_error_type.value,
                    "injection_strategy": agent.injection_strategy.value,
                    "malicious_action_description": agent.description,
                }
                for agent, target in zip(malicious_agents, injection_targets)
            ],
            "num_injected_agents": len(malicious_agents)
        }
        
        return final_output, log

    def run_with_injection(
        self,
        task: Any,
        malicious_agent: FMMaliciousAgent,
        injection_target: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:

        target_role = injection_target['role']
        target_role_index = injection_target.get('role_index', 0)

        # Store the original call_llm method
        original_llm_call = self.dylan_instance.call_llm
        
        current_node_id = None

        # --- Define the FM malicious call_llm method ---
        def fm_malicious_llm_call(*args, **kwargs):
            messages = args[2] if len(args) > 2 else kwargs.get('messages', [])
            

            node_idx = getattr(self.dylan_instance, 'current_node_idx', None)
            round_id = getattr(self.dylan_instance, 'current_round_id', None)
            num_agents = getattr(self.dylan_instance, 'num_agents', None)
            current_role = None
            current_role_index = None
            if node_idx is not None and hasattr(self.dylan_instance, 'nodes'):
                current_role = self.dylan_instance.nodes[node_idx].get('role', None)
                current_role_index = node_idx % num_agents if num_agents else None
            print(f"[FM DyLAN Runner Intercept] Node {node_idx}: role='{current_role}', index={current_role_index}, round={round_id}")

            is_malicious_call = (current_role == target_role and current_role_index == target_role_index)

            if not is_malicious_call:
                return original_llm_call(*args, **kwargs)

            print(f"*** FM Malicious Agent Activated on '{current_role}' (index {current_role_index}, node {node_idx}, round {round_id}) ***")
            
            class MockAgent:
                def __init__(self, role, role_description, dylan_instance):
                    self.role_name = role
                    self.role_type = "DyLAN Agent"
                    self.agent_id = f"dylan_{current_role_index}"
                    self.system_message = role_description
                    self.tool_dict = {}  # DyLAN agents don't have tools
                    self.model_type = "LLM"
                    self.chat_history = []  # Could be populated from previous rounds
                    
                    self.dylan_instance = dylan_instance
                    self.node_id = node_idx
                    self.round = round_id
            
            role_map = self.dylan_instance._get_role_map()
            role_description = role_map.get(current_role, "You are a DyLAN agent.")
            
            mock_agent = MockAgent(current_role, role_description, self.dylan_instance)
            
            agent_context = self.fm_factory.extract_agent_context(
                mock_agent,
                {"name": current_role, "description": f"DyLAN {current_role} agent"}
            )
            
            malicious_agent.agent_context = agent_context
            
            task_input = messages[-1]['content'] if messages else ""
            
            if malicious_agent.injection_strategy == InjectionStrategy.PROMPT_INJECTION:
                # For prompt injection, we need to modify the prompt first
                modified_prompt = self.fm_factory.inject_prompt(task_input, malicious_agent.fm_error_type, malicious_agent.agent_context)
                
                # Create modified messages
                modified_messages = messages.copy()
                if modified_messages and len(modified_messages) > 0:
                    modified_messages[-1] = modified_messages[-1].copy()
                    modified_messages[-1]['content'] = modified_prompt
                
                # Update kwargs with modified messages
                modified_kwargs = kwargs.copy()
                modified_kwargs['messages'] = modified_messages
                
                # Call original function with modified kwargs
                response = original_llm_call(*args, **modified_kwargs)
                
            elif malicious_agent.injection_strategy == InjectionStrategy.RESPONSE_CORRUPTION:
                # For response corruption, call original function first, then corrupt the response
                clean_response = original_llm_call(*args, **kwargs)
                response = self.fm_factory.corrupt_response(clean_response, malicious_agent.fm_error_type, malicious_agent.agent_context)
            else:
                response = original_llm_call(*args, **kwargs)
            
            return response

        # --- Apply the monkey patch ---
        self.dylan_instance.call_llm = fm_malicious_llm_call
        print(f"[FM DyLAN Runner] Monkey-patch applied. Target: role='{target_role}', index={target_role_index}.")
        print(f"[FM DyLAN Runner] Injection: {malicious_agent.fm_error_type.value} via {malicious_agent.injection_strategy.value}")

        # --- Execute the patched workflow ---
        sample = {"query": task.query}
        final_output = self.dylan_instance.inference(sample)

        # --- Cleanup ---
        self.dylan_instance.call_llm = original_llm_call
        print("[FM DyLAN Runner] Original `call_llm` method restored.")
        
        # Collect history from DyLAN's network structure
        full_history = []
        if hasattr(self.dylan_instance, 'nodes'):
            for i, node in enumerate(self.dylan_instance.nodes):
                if node.get('active', False) and node.get('reply'):
                    role = node.get('role', f'agent_{i}')
                    role_index = i % self.dylan_instance.num_agents
                    full_history.append({
                        "role": role,
                        "role_index": role_index,
                        "content": node.get('reply', ''),
                        "node_id": i,
                        "round": i // self.dylan_instance.num_agents
                    })
        
        log = {
            "final_output": final_output,
            "full_history": full_history,
            "injected_role": target_role,
            "injected_role_index": target_role_index,
            "fm_error_type": malicious_agent.fm_error_type.value,
            "injection_strategy": malicious_agent.injection_strategy.value,
            "malicious_action_description": malicious_agent.description,
        }
        return final_output, log 