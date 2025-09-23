# file: agent_systems/fm_agentverse_wrapper.py

from .base_wrapper import SystemWrapper
from malicious_factory import FMMaliciousFactory, FMMaliciousAgent, InjectionStrategy
from typing import Any, Dict, Tuple
from methods import get_method_class

class FMAgentVerseWrapper(SystemWrapper):
    """
    A wrapper for the AgentVerse system using the new FM malicious injection system.
    Supports 28 different malicious injection methods.
    """
    def __init__(self, general_config: Dict[str, Any], method_config: Dict[str, Any]):
        from utils.async_llm import create_llm_instance
        exp_config = general_config.get('experiment_config', {})
        llm_config = exp_config.get('llm_config', {})
        if not llm_config:
            raise ValueError("No LLM configuration found for the specified model name.")
        
        self.llm = create_llm_instance(llm_config)
        method_name = exp_config['system_under_test']['name']  # "agentverse"
        dataset_name = exp_config.get('benchmark_name', None)
        MAS_CLASS = get_method_class(method_name, dataset_name)
        self.agentverse_instance = MAS_CLASS(general_config, method_config_name=None)
        
        self.fm_factory = FMMaliciousFactory(llm=self.llm)
        
        self.call_count = 0
        self.conversation_history = []
        self.current_phase = "init"
        self.role_descriptions = []
        
        print(f"FMAgentVerseWrapper initialized with {MAS_CLASS.__name__}.")

    def run_with_multi_injection(
        self,
        task: Any,
        malicious_agents: list,
        injection_targets: list
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run AgentVerse experiment with multiple malicious agent injections.
        """
        if len(malicious_agents) == 1 and len(injection_targets) == 1:
            # Fall back to single injection for backward compatibility
            return self.run_with_injection(task, malicious_agents[0], injection_targets[0])
        
        # Create a mapping from role to malicious agent
        injection_map = {}
        for agent, target in zip(malicious_agents, injection_targets):
            role = target['role']
            role_index = target.get('role_index', 0)
            key = (role, role_index)
            injection_map[key] = {
                'agent': agent,
                'target': target
            }
        
        print(f"[FMAgentVerseWrapper] Multi-injection targets: {list(injection_map.keys())}")
        
        # Store the original call_llm method
        original_llm_call = self.agentverse_instance.call_llm
        self.call_count = 0
        self.conversation_history = []
        self.current_phase = "init"
        self.role_descriptions = []
        
        def fm_multi_malicious_call_llm(*args, **kwargs):
            self.call_count += 1
            print(f"[FMAgentVerseWrapper Multi] call_llm count: {self.call_count}")
            
            messages = args[2] if len(args) > 2 else kwargs.get('messages', [])
            current_role, current_role_index = self._determine_current_role_and_phase(messages)
            
            self._record_conversation_entry(current_role, current_role_index, messages)
            
            injection_key = (current_role, current_role_index)
            if injection_key not in injection_map:
                response = original_llm_call(*args, **kwargs)
                self._record_response(current_role, current_role_index, response)
                return response
            
            injection_info = injection_map[injection_key]
            malicious_agent = injection_info['agent']
            target = injection_info['target']
            
            print(f"*** FM Multi-Malicious Agent Activated on '{current_role}' (index {current_role_index}) ***")
            print(f"    Error Type: {malicious_agent.fm_error_type.value}, Strategy: {malicious_agent.injection_strategy.value}")
            
            mock_agent = self._create_mock_agent(current_role, current_role_index)
            agent_context = self.fm_factory.extract_agent_context(
                mock_agent,
                {"name": current_role, "description": f"AgentVerse {current_role}"}
            )
            
            malicious_agent.agent_context = agent_context
            
            task_input = messages[-1]['content'] if messages else ""
            
            try:
                if malicious_agent.injection_strategy == InjectionStrategy.PROMPT_INJECTION:
                    modified_prompt = self.fm_factory.inject_prompt(task_input, malicious_agent.fm_error_type, malicious_agent.agent_context)
                    print(f"[FMAgentVerseWrapper Multi] Original prompt length: {len(task_input)}, Modified prompt length: {len(modified_prompt)}")
                    
                    modified_messages = messages.copy()
                    if modified_messages and len(modified_messages) > 0:
                        modified_messages[-1] = modified_messages[-1].copy()
                        modified_messages[-1]['content'] = modified_prompt
                    
                    # Handle the case where we have positional args instead of kwargs
                    if len(args) >= 3:
                        # AgentVerse calls: call_llm(None, None, messages)
                        response = original_llm_call(args[0], args[1], modified_messages)
                    else:
                        modified_kwargs = kwargs.copy()
                        modified_kwargs['messages'] = modified_messages
                        response = original_llm_call(*args, **modified_kwargs)
                    
                elif malicious_agent.injection_strategy == InjectionStrategy.RESPONSE_CORRUPTION:
                    clean_response = original_llm_call(*args, **kwargs)
                    if clean_response:
                        response = self.fm_factory.corrupt_response(clean_response, malicious_agent.fm_error_type, malicious_agent.agent_context)
                    else:
                        print(f"[FMAgentVerseWrapper Multi] Warning: clean_response is None, skipping corruption")
                        response = clean_response
                else:
                    response = original_llm_call(*args, **kwargs)
                
                # Ensure response is not None
                if response is None:
                    print(f"[FMAgentVerseWrapper Multi] Warning: Injection resulted in None response, calling original again")
                    response = original_llm_call(*args, **kwargs)
                    
            except Exception as e:
                print(f"[FMAgentVerseWrapper Multi] Error during injection: {e}, falling back to original call")
                response = original_llm_call(*args, **kwargs)
            
            self._record_response(current_role, current_role_index, response)
            return response
        
        # Apply monkey patch
        self.agentverse_instance.call_llm = fm_multi_malicious_call_llm
        print(f"[FMAgentVerseWrapper] Multi-injection monkey-patch applied for {len(malicious_agents)} targets.")
        
        # Execute with error handling
        sample = {"query": task.query}
        final_output = None
        execution_error = None
        
        try:
            final_output = self.agentverse_instance.inference(sample)
            print("[FMAgentVerseWrapper] Inference completed successfully")
        except Exception as e:
            execution_error = str(e)
            print(f"[FMAgentVerseWrapper] Inference failed due to injection effects: {e}")
            # Create a fallback response indicating the injection was successful in disrupting the system
            final_output = {"response": f"AgentVerse execution disrupted by malicious injection: {execution_error}"}
        
        # Cleanup
        self.agentverse_instance.call_llm = original_llm_call
        print("[FMAgentVerseWrapper] Original `call_llm` method restored.")
        
        # Create comprehensive log
        log = {
            "final_output": final_output,
            "call_count": self.call_count,
            "conversation_history": self.conversation_history,
            "role_descriptions": self.role_descriptions,
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
            "num_injected_agents": len(malicious_agents),
            # Add execution status information for multi-agent injection
            "execution_successful": execution_error is None,
            "execution_error": execution_error,
            "injection_disrupted_system": execution_error is not None,
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
        print(f"[FMAgentVerseWrapper] Target role: {target_role}, index: {target_role_index}")
        
        # Store the original call_llm method
        original_llm_call = self.agentverse_instance.call_llm
        self.call_count = 0
        self.conversation_history = []
        self.current_phase = "init"
        self.role_descriptions = []
        
        def fm_malicious_call_llm(*args, **kwargs):
            self.call_count += 1
            print(f"[FMAgentVerseWrapper] call_llm count: {self.call_count}")
            
            messages = args[2] if len(args) > 2 else kwargs.get('messages', [])
            current_role, current_role_index = self._determine_current_role_and_phase(messages)
            
            self._record_conversation_entry(current_role, current_role_index, messages)
            
            should_inject = (current_role == target_role and current_role_index == target_role_index)
            
            if not should_inject:
                response = original_llm_call(*args, **kwargs)
                self._record_response(current_role, current_role_index, response)
                return response
            
            print(f"*** FM Malicious Agent Activated on '{current_role}' (index {current_role_index}) ***")
            print(f"    Error Type: {malicious_agent.fm_error_type.value}, Strategy: {malicious_agent.injection_strategy.value}")
            
            mock_agent = self._create_mock_agent(current_role, current_role_index)
            agent_context = self.fm_factory.extract_agent_context(
                mock_agent,
                {"name": current_role, "description": f"AgentVerse {current_role}"}
            )
            
            malicious_agent.agent_context = agent_context
            
            task_input = messages[-1]['content'] if messages else ""
            
            try:
                if malicious_agent.injection_strategy == InjectionStrategy.PROMPT_INJECTION:
                    modified_prompt = self.fm_factory.inject_prompt(task_input, malicious_agent.fm_error_type, malicious_agent.agent_context)
                    print(f"[FMAgentVerseWrapper] Original prompt length: {len(task_input)}, Modified prompt length: {len(modified_prompt)}")
                    
                    modified_messages = messages.copy()
                    if modified_messages and len(modified_messages) > 0:
                        modified_messages[-1] = modified_messages[-1].copy()
                        modified_messages[-1]['content'] = modified_prompt
                    
                    # Handle the case where we have positional args instead of kwargs
                    if len(args) >= 3:
                        # AgentVerse calls: call_llm(None, None, messages)
                        response = original_llm_call(args[0], args[1], modified_messages)
                    else:
                        modified_kwargs = kwargs.copy()
                        modified_kwargs['messages'] = modified_messages
                        response = original_llm_call(*args, **modified_kwargs)
                    
                elif malicious_agent.injection_strategy == InjectionStrategy.RESPONSE_CORRUPTION:
                    clean_response = original_llm_call(*args, **kwargs)
                    if clean_response:
                        response = self.fm_factory.corrupt_response(clean_response, malicious_agent.fm_error_type, malicious_agent.agent_context)
                    else:
                        print(f"[FMAgentVerseWrapper] Warning: clean_response is None, skipping corruption")
                        response = clean_response
                else:
                    response = original_llm_call(*args, **kwargs)
                
                # Ensure response is not None
                if response is None:
                    print(f"[FMAgentVerseWrapper] Warning: Injection resulted in None response, calling original again")
                    response = original_llm_call(*args, **kwargs)
                    
            except Exception as e:
                print(f"[FMAgentVerseWrapper] Error during injection: {e}, falling back to original call")
                response = original_llm_call(*args, **kwargs)
            
            self._record_response(current_role, current_role_index, response)
            return response
        
        self.agentverse_instance.call_llm = fm_malicious_call_llm
        print(f"[FMAgentVerseWrapper] Monkey-patch applied. Target: {target_role} (index {target_role_index})")
        print(f"[FMAgentVerseWrapper] Injection: {malicious_agent.fm_error_type.value} via {malicious_agent.injection_strategy.value}")
        
        # Execute with error handling
        sample = {"query": task.query}
        final_output = None
        execution_error = None
        
        try:
            final_output = self.agentverse_instance.inference(sample)
            print("[FMAgentVerseWrapper] Single-agent inference completed successfully")
        except Exception as e:
            execution_error = str(e)
            print(f"[FMAgentVerseWrapper] Single-agent inference failed due to injection effects: {e}")
            # Create a fallback response indicating the injection was successful in disrupting the system
            final_output = {"response": f"AgentVerse execution disrupted by malicious injection: {execution_error}"}
        
        self.agentverse_instance.call_llm = original_llm_call
        print("[FMAgentVerseWrapper] Original `call_llm` method restored.")
        
        log = {
            "final_output": final_output,
            "injected_role": target_role,
            "injected_role_index": target_role_index,
            "call_count": self.call_count,
            "conversation_history": self.conversation_history,
            "role_descriptions": self.role_descriptions,
            "fm_error_type": malicious_agent.fm_error_type.value,
            "injection_strategy": malicious_agent.injection_strategy.value,
            "malicious_action_description": malicious_agent.description,
            # Add execution status information
            "execution_successful": execution_error is None,
            "execution_error": execution_error,
            "injection_disrupted_system": execution_error is not None,
        }
        return final_output, log
    
    def _determine_current_role_and_phase(self, messages: list) -> tuple:
        """
        根据消息内容判断当前的角色和阶段
        AgentVerse的调用顺序：
        1. Role Assigner (分配角色)
        2. Solver (初始解决方案 + 后续修订)
        3. Critics (评审，可能有多个)
        4. Evaluator (最终评估)
        """
        if not messages:
            return "Unknown", 0
        
        content = ""
        for msg in messages:
            content += msg.get('content', '') + " "
        content = content.lower()
        
        if self.call_count == 1:
            self.current_phase = "role_assignment"
            return "RoleAssigner", 0
        
        if any(keyword in content for keyword in ["evaluate", "score", "correctness", "assessment"]):
            self.current_phase = "evaluation"
            return "Evaluator", 0
        
        if any(keyword in content for keyword in ["review", "critic", "feedback", "opinion", "assessment"]):
            critic_index = self._calculate_critic_index()
            self.current_phase = "criticism"
            return "Critic", critic_index
        
        solver_index = self._calculate_solver_index()
        self.current_phase = "solving"
        return "Solver", solver_index
    
    def _calculate_critic_index(self) -> int:
        """计算当前批评者的索引"""
        cnt_agents = getattr(self.agentverse_instance, 'cnt_agents', 3)
        return (self.call_count - 2) % (cnt_agents - 1) if cnt_agents > 1 else 0
    
    def _calculate_solver_index(self) -> int:
        """计算当前解决者的索引（通常只有一个解决者，索引为0）"""
        return 0
    
    def _create_mock_agent(self, role: str, role_index: int):
        """创建模拟的agent对象"""
        class MockAgent:
            def __init__(self, role_name, role_index, agentverse_instance):
                self.role_name = role_name
                self.role_type = f"AgentVerse {role_name}"
                self.agent_id = f"agentverse_{role_name.lower()}_{role_index}"
                self.role_index = role_index
                self.agentverse_instance = agentverse_instance
                self.tool_dict = {}
                self.model_type = "LLM"
                self.chat_history = []
                
                if role_name == "RoleAssigner":
                    self.system_message = "You are a role assigner responsible for assigning appropriate roles to agents based on the given query."
                elif role_name == "Solver":
                    self.system_message = f"You are agent {role_index + 1}, a problem solver responsible for generating solutions."
                elif role_name == "Critic":
                    self.system_message = f"You are agent {role_index + 1}, a critic responsible for reviewing and providing feedback on solutions."
                elif role_name == "Evaluator":
                    self.system_message = "You are an evaluator responsible for assessing the quality and correctness of solutions."
                else:
                    self.system_message = f"You are agent {role_index + 1} in the AgentVerse system."
        
        return MockAgent(role, role_index, self.agentverse_instance)
    
    def _record_conversation_entry(self, role: str, role_index: int, messages: list):
        """记录对话条目"""
        entry = {
            "phase": self.current_phase,
            "role": role,
            "role_index": role_index,
            "call_count": self.call_count,
            "input_messages": messages.copy(),
            "timestamp": "current"
        }
        self.conversation_history.append(entry)
    
    def _record_response(self, role: str, role_index: int, response: str):
        """记录响应"""
        for entry in self.conversation_history:
            if entry["call_count"] == self.call_count:
                entry["response"] = response
                
                if role == "RoleAssigner":
                    try:
                        role_descriptions = self.agentverse_instance.extract_role_descriptions(response)
                        self.role_descriptions = role_descriptions
                        entry["parsed_roles"] = role_descriptions
                    except:
                        pass
                break