# file: agent_systems/fm_dylan_math_wrapper.py

import re
from .base_wrapper import SystemWrapper
from malicious_factory import FMMaliciousFactory, FMMaliciousAgent, FMErrorType, InjectionStrategy
from typing import Any, Dict, Tuple
from methods.dylan.dylan_math import DyLAN_MATH

class FMDyLANMathWrapper(SystemWrapper):
    """
    A wrapper for DyLAN_MATH using the new FM malicious injection system.
    Supports 28 different malicious injection methods.
    """
    def __init__(self, general_config: Dict[str, Any], method_config: Dict[str, Any]):
        from utils.async_llm import create_llm_instance
        exp_config = general_config.get('experiment_config', {})
        llm_config = exp_config.get('llm_config', {})
        if not llm_config:
            raise ValueError("No LLM configuration found for the specified model name.")
        
        self.llm = create_llm_instance(llm_config)
        self.dylan_instance = DyLAN_MATH(general_config, method_config_name=None)
        
        self.fm_factory = FMMaliciousFactory(llm=self.llm)
        
        self.call_count = 0
        self.conversation_history = []
        self.selected_agents = []
        
        print("FMDyLANMathWrapper initialized.")

    def run_with_multi_injection(
        self,
        task: Any,
        malicious_agents: list,
        injection_targets: list
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run DyLAN_MATH experiment with multiple malicious agent injections.
        """
        if len(malicious_agents) == 1 and len(injection_targets) == 1:
            # Fall back to single injection for backward compatibility
            return self.run_with_injection(task, malicious_agents[0], injection_targets[0])
        
        # Create a mapping from target agent index to malicious agent
        injection_map = {}
        for agent, target in zip(malicious_agents, injection_targets):
            # Use call_index if available, otherwise use role_index
            target_agent_index = target.get('call_index', target.get('role_index', 0))
            injection_map[target_agent_index] = {
                'agent': agent,
                'target': target
            }
        
        print(f"[FMDyLANMathWrapper] Multi-injection targets: {list(injection_map.keys())}")
        
        # Store the original call_llm method
        original_llm_call = self.dylan_instance.call_llm
        self.call_count = 0
        self.conversation_history = []
        self.selected_agents = []
        
        def fm_multi_malicious_call_llm(*args, **kwargs):
            self.call_count += 1
            print(f"[FMDyLANMathWrapper Multi] call_llm count: {self.call_count}")
            
            # Record conversation context
            messages = args[2] if len(args) > 2 else kwargs.get('messages', [])
            current_round = self._determine_current_round()
            current_agent_id = self._determine_current_agent_id()
            
            # Record conversation entry
            self._record_conversation_entry(current_round, current_agent_id, messages)
            
            # Determine which agents should be injected
            injected_agents = []
            num_agents = self.dylan_instance.num_agents
            
            # Check agent-based injections for rounds 1, 2, and 4
            if self.call_count <= 2 * num_agents:
                # Rounds 1 and 2: direct agent index mapping
                current_agent_index = (self.call_count - 1) % num_agents
                if current_agent_index in injection_map:
                    injected_agents.append((current_agent_index, injection_map[current_agent_index]))
            elif self.call_count == 2 * num_agents + 1:
                # Round 3: ranking call - check for aggregator injection
                aggregator_index = num_agents  # Use num_agents as aggregator index
                if aggregator_index in injection_map:
                    injected_agents.append((aggregator_index, injection_map[aggregator_index]))
            else:
                # Round 4: final debate with selected agents
                remaining_calls = self.call_count - (2 * num_agents + 1)
                if remaining_calls <= len(self.selected_agents):
                    current_selected_agent_index = remaining_calls - 1
                    if current_selected_agent_index < len(self.selected_agents):
                        selected_agent_index = self.selected_agents[current_selected_agent_index]
                        if selected_agent_index in injection_map:
                            injected_agents.append((selected_agent_index, injection_map[selected_agent_index]))
            
            # Apply injections if any agents should be injected
            if injected_agents:
                # For simplicity, if multiple agents should be injected at the same call,
                # we apply the first one. In practice, this should be rare.
                target_agent_index, injection_info = injected_agents[0]
                malicious_agent = injection_info['agent']
                target = injection_info['target']
                
                # Check if this is an aggregator injection
                is_aggregator = target_agent_index == num_agents
                
                if is_aggregator:
                    print(f"*** FM Multi-Malicious Aggregator Activated on call_llm #{self.call_count} (ConsensusAggregator) ***")
                    print(f"    Error Type: {malicious_agent.fm_error_type.value}, Strategy: {malicious_agent.injection_strategy.value}")
                else:
                    print(f"*** FM Multi-Malicious Agent Activated on call_llm #{self.call_count} (Agent {target_agent_index + 1}) ***")
                    print(f"    Error Type: {malicious_agent.fm_error_type.value}, Strategy: {malicious_agent.injection_strategy.value}")
                
                # Extract agent context - same logic as single injection
                class MockAgent:
                    def __init__(self, role, role_description, dylan_instance, agent_index, is_aggregator=False):
                        self.role_name = role
                        self.role_type = "DyLAN Math Agent" if not is_aggregator else "DyLAN Math Aggregator"
                        self.agent_id = f"dylan_math_{agent_index}" if not is_aggregator else "dylan_math_aggregator"
                        self.system_message = role_description
                        self.tool_dict = {}
                        self.model_type = "LLM"
                        self.chat_history = []
                        
                        self.dylan_instance = dylan_instance
                        self.agent_index = agent_index
                        self.round = current_round
                
                if is_aggregator:
                    role_description = "You are a DyLAN Math Consensus Aggregator responsible for ranking and selecting the best mathematical solutions from other agents."
                    mock_agent = MockAgent("ConsensusAggregator", role_description, self.dylan_instance, target_agent_index, is_aggregator=True)
                else:
                    role_description = self.dylan_instance.system_prompt
                    mock_agent = MockAgent(current_agent_id, role_description, self.dylan_instance, target_agent_index)
                
                agent_context = self.fm_factory.extract_agent_context(
                    mock_agent,
                    {"name": current_agent_id, "description": f"DyLAN Math {current_agent_id}"}
                )
                
                # Update malicious_agent's context
                malicious_agent.agent_context = agent_context
                
                # Apply injection based on strategy
                task_input = messages[-1]['content'] if messages else ""
                
                if malicious_agent.injection_strategy == InjectionStrategy.PROMPT_INJECTION:
                    modified_prompt = self.fm_factory.inject_prompt(task_input, malicious_agent.fm_error_type, malicious_agent.agent_context)
                    
                    modified_messages = messages.copy()
                    if modified_messages and len(modified_messages) > 0:
                        modified_messages[-1] = modified_messages[-1].copy()
                        modified_messages[-1]['content'] = modified_prompt
                    
                    modified_kwargs = kwargs.copy()
                    modified_kwargs['messages'] = modified_messages
                    
                    response = original_llm_call(*args, **modified_kwargs)
                    
                elif malicious_agent.injection_strategy == InjectionStrategy.RESPONSE_CORRUPTION:
                    clean_response = original_llm_call(*args, **kwargs)
                    response = self.fm_factory.corrupt_response(clean_response, malicious_agent.fm_error_type, malicious_agent.agent_context)
                else:
                    response = original_llm_call(*args, **kwargs)
                
                self._record_response(current_round, current_agent_id, response)
                return response
            
            # No injection for this call
            response = original_llm_call(*args, **kwargs)
            self._record_response(current_round, current_agent_id, response)
            
            # Parse ranking result for round 3
            num_agents = self.dylan_instance.num_agents
            if self.call_count == 2 * num_agents + 1:
                self._parse_ranking_result(response)
            
            return response
        
        # Apply monkey patch
        self.dylan_instance.call_llm = fm_multi_malicious_call_llm
        print(f"[FMDyLANMathWrapper] Multi-injection monkey-patch applied for {len(malicious_agents)} targets.")
        
        # Execute
        sample = {"query": task.query}
        final_output = self.dylan_instance.inference(sample)
        
        # Cleanup
        self.dylan_instance.call_llm = original_llm_call
        print("[FMDyLANMathWrapper] Original `call_llm` method restored.")
        
        # Create comprehensive log
        log = {
            "final_output": final_output,
            "call_count": self.call_count,
            "conversation_history": self.conversation_history,
            "selected_agents": self.selected_agents,
            "selected_agent_names": [f"Assistant {i+1}" for i in self.selected_agents],
            "multi_injection_info": [
                {
                    "injected_agent_index": target.get('call_index', target.get('role_index', 0)),
                    "injected_agent_id": f"Assistant {target.get('call_index', target.get('role_index', 0)) + 1}",
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
        target_agent_index = injection_target.get('call_index', injection_target.get('role_index', 0))
        print(f"[FMDyLANMathWrapper] Target agent index: {target_agent_index} (from injection_target: {injection_target})")
        
        # Store the original call_llm method
        original_llm_call = self.dylan_instance.call_llm
        self.call_count = 0
        self.conversation_history = []
        self.selected_agents = []
        
        def fm_malicious_call_llm(*args, **kwargs):
            self.call_count += 1
            print(f"[FMDyLANMathWrapper] call_llm count: {self.call_count}")
            
            messages = args[2] if len(args) > 2 else kwargs.get('messages', [])
            current_round = self._determine_current_round()
            current_agent_id = self._determine_current_agent_id()
            
            self._record_conversation_entry(current_round, current_agent_id, messages)
            
            should_inject = False
            
            if messages and len(messages) > 0:
                system_message = messages[0].get('content', '') if messages[0].get('role') == 'system' else ''
                
                num_agents = self.dylan_instance.num_agents
                if self.call_count <= 2 * num_agents:
                    current_agent_index = (self.call_count - 1) % num_agents
                    should_inject = (current_agent_index == target_agent_index)
                
                elif self.call_count == 2 * num_agents + 1:
                    should_inject = False
                
                else:
                    remaining_calls = self.call_count - (2 * num_agents + 1)
                    if remaining_calls <= len(self.selected_agents):
                        current_selected_agent_index = remaining_calls - 1
                        if current_selected_agent_index < len(self.selected_agents):
                            selected_agent_index = self.selected_agents[current_selected_agent_index]
                            should_inject = (selected_agent_index == target_agent_index)
            
            if should_inject:
                print(f"*** FM Malicious Agent Activated on call_llm #{self.call_count} (Agent {target_agent_index + 1}) ***")
                
                class MockAgent:
                    def __init__(self, role, role_description, dylan_instance, agent_index):
                        self.role_name = role
                        self.role_type = "DyLAN Math Agent"
                        self.agent_id = f"dylan_math_{agent_index}"
                        self.system_message = role_description
                        self.tool_dict = {}  # DyLAN agents don't have tools
                        self.model_type = "LLM"
                        self.chat_history = []  # Could be populated from previous rounds
                        
                        self.dylan_instance = dylan_instance
                        self.agent_index = agent_index
                        self.round = current_round
                
                role_description = self.dylan_instance.system_prompt
                
                mock_agent = MockAgent(current_agent_id, role_description, self.dylan_instance, target_agent_index)
                
                agent_context = self.fm_factory.extract_agent_context(
                    mock_agent,
                    {"name": current_agent_id, "description": f"DyLAN Math {current_agent_id}"}
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
                
                self._record_response(current_round, current_agent_id, response)
                return response
            
            response = original_llm_call(*args, **kwargs)
            
            self._record_response(current_round, current_agent_id, response)
            
            num_agents = self.dylan_instance.num_agents
            if self.call_count == 2 * num_agents + 1:
                self._parse_ranking_result(response)
            
            return response
        
        self.dylan_instance.call_llm = fm_malicious_call_llm
        print(f"[FMDyLANMathWrapper] Monkey-patch applied. Target: Agent {target_agent_index + 1} (will be injected when called).")
        print(f"[FMDyLANMathWrapper] Injection: {malicious_agent.fm_error_type.value} via {malicious_agent.injection_strategy.value}")
        
        sample = {"query": task.query}
        final_output = self.dylan_instance.inference(sample)
        
        self.dylan_instance.call_llm = original_llm_call
        print("[FMDyLANMathWrapper] Original `call_llm` method restored.")
        
        log = {
            "final_output": final_output,
            "malicious_action_description": malicious_agent.description,
            "injected_agent_index": target_agent_index,
            "injected_agent_id": f"Assistant {target_agent_index + 1}",
            "call_count": self.call_count,
            "conversation_history": self.conversation_history,
            "selected_agents": self.selected_agents,
            "selected_agent_names": [f"Assistant {i+1}" for i in self.selected_agents],
            "fm_error_type": malicious_agent.fm_error_type.value,
            "injection_strategy": malicious_agent.injection_strategy.value,
        }
        return final_output, log
    
    def _parse_ranking_result(self, ranking_response: str):
        """解析第3轮的排名结果"""
        import re
        
        print(f"[FMDyLANMathWrapper] Parsing ranking result: {ranking_response[:100]}...")
        
        pattern = r'\[([1234]),\s*([1234])\]'
        matches = re.findall(pattern, ranking_response)
        
        try:
            if matches:
                match = matches[-1]
                tops = [int(match[0])-1, int(match[1])-1]
                
                def clip(x):
                    if x < 0:
                        return 0
                    if x > 3:
                        return 3
                    return x
                
                tops = [clip(x) for x in tops]
                self.selected_agents = tops
                print(f"[FMDyLANMathWrapper] Successfully parsed ranking result: selected agents {[i+1 for i in tops]}")
            else:
                self.selected_agents = [0, 1]
                print(f"[FMDyLANMathWrapper] No ranking pattern found, using default: selected agents {[i+1 for i in self.selected_agents]}")
        except Exception as e:
            self.selected_agents = [0, 1]
            print(f"[FMDyLANMathWrapper] Ranking parsing failed: {e}, using default: selected agents {[i+1 for i in self.selected_agents]}")
    
    def _determine_current_round(self) -> str:
        """根据 call_count 确定当前轮次"""
        num_agents = self.dylan_instance.num_agents
        
        if self.call_count <= num_agents:
            return "Round 1 - Initial Solutions"
        elif self.call_count <= 2 * num_agents:
            return "Round 2 - Debate"
        elif self.call_count == 2 * num_agents + 1:
            return "Round 3 - Ranking"
        else:
            return "Round 4 - Final Debate"
    
    def _determine_current_agent_id(self) -> str:
        """根据 call_count 确定当前 agent ID"""
        num_agents = self.dylan_instance.num_agents
        
        if self.call_count <= num_agents:
            agent_index = self.call_count - 1
            return f"Assistant {agent_index + 1}"
        elif self.call_count <= 2 * num_agents:
            agent_index = self.call_count - num_agents - 1
            return f"Assistant {agent_index + 1}"
        elif self.call_count == 2 * num_agents + 1:
            return "Ranker"
        else:
            remaining_calls = self.call_count - (2 * num_agents + 1)
            if remaining_calls <= len(self.selected_agents):
                selected_agent_index = self.selected_agents[remaining_calls - 1]
                return f"Selected Assistant {selected_agent_index + 1}"
            else:
                return f"Selected Assistant {remaining_calls}"
    
    def _record_conversation_entry(self, round_name: str, agent_id: str, messages: list):
        """记录对话条目"""
        entry = {
            "round": round_name,
            "agent_id": agent_id,
            "call_count": self.call_count,
            "input_messages": messages.copy(),
            "timestamp": "current"
        }
        self.conversation_history.append(entry)
    
    def _record_response(self, round_name: str, agent_id: str, response: str):
        """记录响应"""
        for entry in self.conversation_history:
            if entry["call_count"] == self.call_count:
                entry["response"] = response
                break 