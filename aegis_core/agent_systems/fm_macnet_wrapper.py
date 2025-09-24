# file: agent_systems/fm_macnet_wrapper.py

import re
import asyncio
from .base_wrapper import SystemWrapper
from malicious_factory import FMMaliciousFactory, FMMaliciousAgent, FMErrorType, InjectionStrategy
from typing import Any, Dict, Tuple

from methods import get_method_class

class FMMacNetWrapper(SystemWrapper):
    """
    A wrapper for the MacNet system using the new FM malicious injection system.
    Supports 28 different malicious injection methods across different MacNet nodes.
    """
    def __init__(self, general_config: Dict[str, Any], method_config: Dict[str, Any]):
        from utils.async_llm import create_llm_instance
        exp_config = general_config.get('experiment_config', {})
        llm_config = exp_config.get('llm_config', {})
        if not llm_config:
            raise ValueError("No LLM configuration found for the specified model name.")

        self.llm = create_llm_instance(llm_config)
        
        method_name = exp_config['system_under_test']['name']  # "macnet"
        dataset_name = exp_config.get('benchmark_name', None)
        MAS_CLASS = get_method_class(method_name, dataset_name)
        self.macnet_instance = MAS_CLASS(general_config, method_config_name=None)

        self.fm_factory = FMMaliciousFactory(llm=self.llm)

        print(f"FMMacNetWrapper initialized with {MAS_CLASS.__name__}.")

    def run_with_multi_injection(
        self,
        task: Any,
        malicious_agents: list,
        injection_targets: list
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run MacNet experiment with multiple malicious agent injections.
        """
        if len(malicious_agents) == 1 and len(injection_targets) == 1:
            # Fall back to single injection for backward compatibility
            return self.run_with_injection(task, malicious_agents[0], injection_targets[0])
        
        # Create a mapping from node_id to malicious agent
        injection_map = {}
        for agent, target in zip(malicious_agents, injection_targets):
            node_id = target.get('node_id', target.get('role_index', 0))
            injection_map[node_id] = {
                'agent': agent,
                'target': target
            }
        
        # Store the original call_llm method
        original_llm_call = self.macnet_instance.call_llm
        
        # Track all LLM calls for logging
        llm_call_history = []
        call_count = 0
        
        # --- Define the multi-target FM malicious call_llm method ---
        def fm_multi_malicious_llm_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Extract arguments - MacNet uses: call_llm(prompt=..., system_prompt=..., temperature=...)
            prompt = kwargs.get('prompt', args[0] if len(args) > 0 else None)
            system_prompt = kwargs.get('system_prompt', args[1] if len(args) > 1 else None)
            temperature = kwargs.get('temperature', args[2] if len(args) > 2 else None)
            
            # Determine current node from call context (now passed directly from MacNet)
            current_node_id = self._get_current_node_id(prompt, system_prompt, kwargs)
            
            print(f"[FM Multi-MacNet Runner Intercept] Call #{call_count}, Node ID: {current_node_id}")
            
            # Check if this node should be injected
            if current_node_id not in injection_map:
                clean_kwargs = {k: v for k, v in kwargs.items() if k != 'node_id'}
                response = original_llm_call(*args, **clean_kwargs)
                # Log normal call
                llm_call_history.append({
                    "call_id": call_count,
                    "node_id": current_node_id,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "response": response,
                    "injected": False
                })
                return response
            
            # Get the malicious agent and target for this injection
            injection_info = injection_map[current_node_id]
            malicious_agent = injection_info['agent']
            target = injection_info['target']
            
            print(f"*** FM Multi-Malicious MacNet Node Activated on Node {current_node_id} ***")
            print(f"*** Type: {malicious_agent.fm_error_type.value}, Strategy: {malicious_agent.injection_strategy.value} ***")
            
            # Create mock agent for this node
            mock_agent = MockMacNetNode(current_node_id, self.macnet_instance)
            
            # Extract agent context and set it for malicious agent
            agent_context = self.fm_factory.extract_agent_context(
                mock_agent,
                {"name": mock_agent.role_name, "description": mock_agent.system_message}
            )
            malicious_agent.agent_context = agent_context
            
            # Perform FM injection
            if malicious_agent.injection_strategy == InjectionStrategy.PROMPT_INJECTION:
                # Modify the input prompt
                modified_prompt = self.fm_factory.inject_prompt(
                    prompt, malicious_agent.fm_error_type, malicious_agent.agent_context
                )
                modified_kwargs = kwargs.copy()
                modified_kwargs['prompt'] = modified_prompt
                # Remove node_id from kwargs before calling original method
                clean_modified_kwargs = {k: v for k, v in modified_kwargs.items() if k != 'node_id'}
                response = original_llm_call(*args, **clean_modified_kwargs)
                
            elif malicious_agent.injection_strategy == InjectionStrategy.RESPONSE_CORRUPTION:
                # Get clean response first, then corrupt it
                # Remove node_id from kwargs before calling original method
                clean_kwargs = {k: v for k, v in kwargs.items() if k != 'node_id'}
                clean_response = original_llm_call(*args, **clean_kwargs)
                response = self.fm_factory.corrupt_response(
                    clean_response, malicious_agent.fm_error_type, malicious_agent.agent_context
                )
            else:
                # Fallback to original response
                # Remove node_id from kwargs before calling original method
                clean_kwargs = {k: v for k, v in kwargs.items() if k != 'node_id'}
                response = original_llm_call(*args, **clean_kwargs)
            
            # Log injected call
            llm_call_history.append({
                "call_id": call_count,
                "node_id": current_node_id,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response": response,
                "injected": True,
                "fm_error_type": malicious_agent.fm_error_type.value,
                "injection_strategy": malicious_agent.injection_strategy.value
            })
            
            return response
        
        # --- Monkey patch ---
        self.macnet_instance.call_llm = fm_multi_malicious_llm_call
        print(f"[FM Multi-MacNet Runner] Monkey-patch applied to call_llm method.")
        
        # Run the experiment
        sample = {"query": task.query}
        final_output = self.macnet_instance.inference(sample)
        
        # Restore original method
        self.macnet_instance.call_llm = original_llm_call
        print("[FM Multi-MacNet Runner] Original `call_llm` method restored.")
        
        # Build comprehensive log
        log = {
            "final_output": final_output,
            "llm_call_history": llm_call_history,
            "injected_nodes": list(injection_map.keys()),
            "total_llm_calls": call_count,
            "graph_topology": getattr(self.macnet_instance, 'topology', None),
            "nodes_info": self._get_nodes_info()
        }
        
        return final_output, log

    def run_with_injection(
        self,
        task: Any,
        malicious_agent: FMMaliciousAgent,
        injection_target: dict
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run MacNet experiment with single malicious agent injection.
        """
        target_node_id = injection_target.get('node_id', injection_target.get('role_index', 0))
        
        # Store the original call_llm method from MAS base class
        # MacNet Node calls self.env.call_llm, where env is MacNet_Main instance
        original_llm_call = self.macnet_instance.call_llm
        
        # Track all LLM calls for logging
        llm_call_history = []
        call_count = 0
        
        # --- Define the FM malicious call_llm method ---
        def fm_malicious_llm_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Extract arguments
            prompt = kwargs.get('prompt', args[0] if len(args) > 0 else None)
            system_prompt = kwargs.get('system_prompt', args[1] if len(args) > 1 else None)
            temperature = kwargs.get('temperature', args[2] if len(args) > 2 else None)
            
            # Determine current node (now passed directly from MacNet)
            current_node_id = self._get_current_node_id(prompt, system_prompt, kwargs)
            
            print(f"[FM MacNet Runner Intercept] Call #{call_count}, Node: {current_node_id}, Target: {target_node_id}")
            
            # Check if this is the target node
            if current_node_id != target_node_id:
                # Remove node_id from kwargs before calling original method
                clean_kwargs = {k: v for k, v in kwargs.items() if k != 'node_id'}
                response = original_llm_call(*args, **clean_kwargs)
                llm_call_history.append({
                    "call_id": call_count,
                    "node_id": current_node_id,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "response": response,
                    "injected": False
                })
                return response
            
            print(f"*** FM Malicious MacNet Node Activated on Node {current_node_id} ***")
            print(f"*** Type: {malicious_agent.fm_error_type.value}, Strategy: {malicious_agent.injection_strategy.value} ***")
            
            # Create mock agent for this node
            mock_agent = MockMacNetNode(current_node_id, self.macnet_instance)
            
            # Extract agent context and set it for malicious agent
            agent_context = self.fm_factory.extract_agent_context(
                mock_agent,
                {"name": mock_agent.role_name, "description": mock_agent.system_message}
            )
            malicious_agent.agent_context = agent_context
            
            # Perform FM injection
            if malicious_agent.injection_strategy == InjectionStrategy.PROMPT_INJECTION:
                modified_prompt = self.fm_factory.inject_prompt(
                    prompt, malicious_agent.fm_error_type, malicious_agent.agent_context
                )
                modified_kwargs = kwargs.copy()
                modified_kwargs['prompt'] = modified_prompt
                # Remove node_id from kwargs before calling original method
                clean_modified_kwargs = {k: v for k, v in modified_kwargs.items() if k != 'node_id'}
                response = original_llm_call(*args, **clean_modified_kwargs)
                
            elif malicious_agent.injection_strategy == InjectionStrategy.RESPONSE_CORRUPTION:
                # Remove node_id from kwargs before calling original method
                clean_kwargs = {k: v for k, v in kwargs.items() if k != 'node_id'}
                clean_response = original_llm_call(*args, **clean_kwargs)
                response = self.fm_factory.corrupt_response(
                    clean_response, malicious_agent.fm_error_type, malicious_agent.agent_context
                )
            else:
                # Remove node_id from kwargs before calling original method
                clean_kwargs = {k: v for k, v in kwargs.items() if k != 'node_id'}
                response = original_llm_call(*args, **clean_kwargs)
            
            llm_call_history.append({
                "call_id": call_count,
                "node_id": current_node_id,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response": response,
                "injected": True,
                "fm_error_type": malicious_agent.fm_error_type.value,
                "injection_strategy": malicious_agent.injection_strategy.value
            })
            
            return response
        
        # --- Monkey patch ---
        self.macnet_instance.call_llm = fm_malicious_llm_call
        print(f"[FM MacNet Runner] Monkey-patch applied. Target Node: {target_node_id}.")
        print(f"[FM MacNet Runner] Injection: {malicious_agent.fm_error_type.value} via {malicious_agent.injection_strategy.value}")
        
        # Run the experiment
        sample = {"query": task.query}
        final_output = self.macnet_instance.inference(sample)
        
        # Restore original method
        self.macnet_instance.call_llm = original_llm_call
        print("[FM MacNet Runner] Original `call_llm` method restored.")
        
        # Build log
        log = {
            "final_output": final_output,
            "llm_call_history": llm_call_history,
            "injected_node_id": target_node_id,
            "fm_error_type": malicious_agent.fm_error_type.value,
            "injection_strategy": malicious_agent.injection_strategy.value,
            "malicious_action_description": malicious_agent.description,
            "total_llm_calls": call_count,
            "graph_topology": getattr(self.macnet_instance, 'topology', None),
            "nodes_info": self._get_nodes_info()
        }
        
        return final_output, log

    def _get_current_node_id(self, prompt, system_prompt, kwargs):
        """
        Get the current node ID directly from kwargs (passed by MacNet).
        """
        # Method 1: Direct node_id from kwargs (most reliable)
        if 'node_id' in kwargs:
            node_id = kwargs['node_id']
            print(f"[DEBUG] Got node_id from kwargs: {node_id}")
            return node_id
        
        # Method 2: Fallback to original detection methods
        print(f"[DEBUG] No node_id in kwargs, using fallback")
        return 0

    def _get_nodes_info(self):
        """
        Get information about all nodes in the MacNet graph.
        """
        nodes_info = {}
        if hasattr(self.macnet_instance, 'nodes'):
            for node_id, node in self.macnet_instance.nodes.items():
                nodes_info[node_id] = {
                    "node_id": node_id,
                    "depth": getattr(node, 'depth', 0),
                    "temperature": getattr(node, 'temperature', 0.2),
                    "system_message": getattr(node, 'system_message', '')
                }
        return nodes_info

class MockMacNetNode:
    """
    Mock node class for MacNet to work with FM factory.
    """
    def __init__(self, node_id, macnet_instance):
        self.node_id = node_id
        self.role_name = f"Node {node_id}"
        self.role_type = "MacNet Node"
        self.agent_id = f"macnet_node_{node_id}"
        self.macnet_instance = macnet_instance
        
        # Get node-specific information if available
        if hasattr(macnet_instance, 'nodes') and node_id in macnet_instance.nodes:
            node = macnet_instance.nodes[node_id]
            self.system_message = getattr(node, 'system_message', self._get_default_system_message())
            self.temperature = getattr(node, 'temperature', 0.2)
        else:
            self.system_message = self._get_default_system_message()
            self.temperature = 0.2

    def _get_default_system_message(self):
        """
        Get default system message for MacNet nodes.
        """
        return f"You are Node {self.node_id} in a MacNet multi-agent collaboration system. Your role is to process information and contribute to the collective problem-solving process."