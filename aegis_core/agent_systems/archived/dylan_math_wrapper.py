# file: agent_systems/dylan_math_wrapper.py

import re
from .base_wrapper import SystemWrapper
from malicious_factory.agent import MaliciousAgent
from typing import Any, Dict, Tuple
from methods.dylan.dylan_math import DyLAN_MATH

class DyLANMathWrapper(SystemWrapper):
    """
    A wrapper for DyLAN_MATH. 注入逻辑基于 call_llm 调用次数。
    """
    def __init__(self, general_config: Dict[str, Any], method_config: Dict[str, Any]):
        from utils.async_llm import create_llm_instance
        exp_config = general_config.get('experiment_config', {})
        llm_config = exp_config.get('llm_config', {})
        if not llm_config:
            raise ValueError("No LLM configuration found for the specified model name.")
        self.llm = create_llm_instance(llm_config)
        self.dylan_instance = DyLAN_MATH(general_config, method_config_name=None)
        self.call_count = 0
        self.conversation_history = []  # 记录完整的对话历史
        self.selected_agents = []  # 记录第3轮选中的 agent
        print("DyLANMathWrapper initialized.")

    def run_with_injection(
        self,
        task: Any,
        malicious_agent: MaliciousAgent,
        injection_target: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        # 兼容 role_index 和 call_index，优先使用 call_index
        target_agent_index = injection_target.get('call_index', injection_target.get('role_index', 0))
        print(f"[DyLANMathWrapper] Target agent index: {target_agent_index} (from injection_target: {injection_target})")
        
        # Store the original call_llm method
        original_llm_call = self.dylan_instance.call_llm
        self.call_count = 0
        self.conversation_history = []  # 重置对话历史
        self.selected_agents = []  # 记录第3轮选中的 agent
        
        # 创建 factory 实例用于注入逻辑
        from malicious_factory import MaliciousAgentFactory
        factory = MaliciousAgentFactory(llm=self.llm)
        
        def malicious_call_llm(*args, **kwargs):
            self.call_count += 1
            print(f"[DyLANMathWrapper] call_llm count: {self.call_count}")
            
            # 记录当前调用的上下文信息
            messages = args[2] if len(args) > 2 else kwargs.get('messages', [])
            current_round = self._determine_current_round()
            current_agent_id = self._determine_current_agent_id()
            
            # 记录对话历史
            self._record_conversation_entry(current_round, current_agent_id, messages)
            
            # 判断是否应该注入恶意内容
            should_inject = False
            
            # 通过检查 messages 中的系统提示来判断是否是特定 agent 的调用
            if messages and len(messages) > 0:
                system_message = messages[0].get('content', '') if messages[0].get('role') == 'system' else ''
                
                # 第1轮和第2轮：通过 call_count 判断 agent 索引
                num_agents = self.dylan_instance.num_agents
                if self.call_count <= 2 * num_agents:
                    current_agent_index = (self.call_count - 1) % num_agents
                    should_inject = (current_agent_index == target_agent_index)
                
                # 第3轮：排名调用，解析结果
                elif self.call_count == 2 * num_agents + 1:
                    should_inject = False
                    # 这里不注入，但需要解析排名结果
                    # 我们会在响应记录后解析
                
                # 第4轮：最终辩论，检查目标 agent 是否被选中
                else:
                    # 使用解析出的排名结果
                    remaining_calls = self.call_count - (2 * num_agents + 1)
                    if remaining_calls <= len(self.selected_agents):
                        # 检查当前调用对应的 agent 是否是目标 agent
                        current_selected_agent_index = remaining_calls - 1
                        if current_selected_agent_index < len(self.selected_agents):
                            selected_agent_index = self.selected_agents[current_selected_agent_index]
                            should_inject = (selected_agent_index == target_agent_index)
            
            if should_inject:
                print(f"*** Malicious Agent Activated on call_llm #{self.call_count} (Agent {target_agent_index + 1}) ***")
                
                # 使用 factory 的统一注入逻辑
                task_input = messages[-1]['content'] if messages else ""
                response = factory.inject_malicious_behavior(
                    lambda: original_llm_call(*args, **kwargs),
                    malicious_agent,
                    task_input=task_input,
                    messages=messages
                )
                
                # 记录恶意注入的响应
                self._record_response(current_round, current_agent_id, response)
                return response
            
            response = original_llm_call(*args, **kwargs)
            
            # 记录响应
            self._record_response(current_round, current_agent_id, response)
            
            # 第3轮：解析排名结果
            if self.call_count == 2 * num_agents + 1:
                self._parse_ranking_result(response)
            
            return response
        
        self.dylan_instance.call_llm = malicious_call_llm
        print(f"[DyLANMathWrapper] Monkey-patch applied. Target: Agent {target_agent_index + 1} (will be injected when called).")
        sample = {"query": task.query}
        final_output = self.dylan_instance.inference(sample)
        self.dylan_instance.call_llm = original_llm_call
        print("[DyLANMathWrapper] Original `call_llm` method restored.")
        log = {
            "final_output": final_output,
            "malicious_action_description": malicious_agent.description,
            "injected_agent_index": target_agent_index,
            "injected_agent_id": f"Assistant {target_agent_index + 1}",
            "call_count": self.call_count,
            "conversation_history": self.conversation_history,
            "selected_agents": self.selected_agents,
            "selected_agent_names": [f"Assistant {i+1}" for i in self.selected_agents]
        }
        return final_output, log
    
    def _parse_ranking_result(self, ranking_response: str):
        """解析第3轮的排名结果"""
        import re
        
        print(f"[DyLANMathWrapper] Parsing ranking result: {ranking_response[:100]}...")
        
        # 使用与 dylan_math.py 相同的解析逻辑
        pattern = r'\[([1234]),\s*([1234])\]'
        matches = re.findall(pattern, ranking_response)
        
        try:
            if matches:
                match = matches[-1]  # 取最后一个匹配
                tops = [int(match[0])-1, int(match[1])-1]  # 转换为0-based索引
                
                # 边界检查
                def clip(x):
                    if x < 0:
                        return 0
                    if x > 3:
                        return 3
                    return x
                
                tops = [clip(x) for x in tops]
                self.selected_agents = tops
                print(f"[DyLANMathWrapper] Successfully parsed ranking result: selected agents {[i+1 for i in tops]}")
            else:
                # 如果没有找到匹配，使用默认值
                self.selected_agents = [0, 1]
                print(f"[DyLANMathWrapper] No ranking pattern found, using default: selected agents {[i+1 for i in self.selected_agents]}")
        except Exception as e:
            # 解析失败，使用默认值
            self.selected_agents = [0, 1]
            print(f"[DyLANMathWrapper] Ranking parsing failed: {e}, using default: selected agents {[i+1 for i in self.selected_agents]}")
    
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
            # 第一轮：每个 agent 独立回答
            agent_index = self.call_count - 1
            return f"Assistant {agent_index + 1}"
        elif self.call_count <= 2 * num_agents:
            # 第二轮：每个 agent 辩论
            agent_index = self.call_count - num_agents - 1
            return f"Assistant {agent_index + 1}"
        elif self.call_count == 2 * num_agents + 1:
            # 第三轮：排名（只有一个调用）
            return "Ranker"
        else:
            # 第四轮：最终辩论（只有选中的 agent）
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
        # 找到对应的对话条目并添加响应
        for entry in self.conversation_history:
            if entry["call_count"] == self.call_count:
                entry["response"] = response
                break 