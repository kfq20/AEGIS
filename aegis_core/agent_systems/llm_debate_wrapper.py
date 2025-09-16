# file: agent_systems/llm_debate_wrapper.py

from .base_wrapper import SystemWrapper
from malicious_factory.agent import MaliciousAgent
from typing import Any, Dict, Tuple
from methods import get_method_class

class LLM_Debate_Wrapper(SystemWrapper):
    """
    A wrapper for the LLM Debate system. Supports malicious agent injection via factory.
    """
    def __init__(self, general_config: Dict[str, Any], method_config: Dict[str, Any]):
        from utils.async_llm import create_llm_instance
        exp_config = general_config.get('experiment_config', {})
        llm_config = exp_config.get('llm_config', {})
        if not llm_config:
            raise ValueError("No LLM configuration found for the specified model name.")
        self.llm = create_llm_instance(llm_config)
        method_name = exp_config['system_under_test']['name']  # "llm_debate"
        dataset_name = exp_config.get('benchmark_name', None)
        MAS_CLASS = get_method_class(method_name, dataset_name)
        self.debate_instance = MAS_CLASS(general_config, method_config_name=None)
        print(f"LLM_Debate_Wrapper initialized with {MAS_CLASS.__name__}.")

    def run_with_injection(
        self,
        task: Any,
        malicious_agent: MaliciousAgent,
        injection_target: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        # 只支持按 agent index 注入
        target_agent_index = injection_target.get('role_index', 0)
        print(f"[LLM_Debate_Wrapper] Target agent index: {target_agent_index}")
        original_call_llm = self.debate_instance.call_llm
        agents_num = getattr(self.debate_instance, 'agents_num', 3)
        rounds_num = getattr(self.debate_instance, 'rounds_num', 2)
        call_count = {'count': 0}
        # 创建 factory 实例
        from malicious_factory import MaliciousAgentFactory
        factory = MaliciousAgentFactory(llm=self.llm)
        # --- 定义恶意 call_llm ---
        def malicious_call_llm(*args, **kwargs):
            idx = call_count['count'] % agents_num
            call_count['count'] += 1
            if idx == target_agent_index:
                print(f"*** Malicious Agent Activated on index {idx} [Method {malicious_agent.method}] ***")
                
                # 从 messages 中提取 task_input
                messages = kwargs.get('messages', [])
                task_input = ""
                if messages:
                    # 获取最后一条 user 消息作为当前任务
                    for msg in reversed(messages):
                        if msg.get('role') == 'user':
                            task_input = msg.get('content', '')
                            break
                
                return factory.inject_malicious_behavior(
                    lambda: original_call_llm(*args, **kwargs),
                    malicious_agent,
                    task_input=task_input,
                    messages=messages
                )
            return original_call_llm(*args, **kwargs)
        # --- Monkey patch ---
        self.debate_instance.call_llm = malicious_call_llm
        print(f"[LLM_Debate_Wrapper] Monkey-patch applied. Target index: {target_agent_index}.")
        sample = {"query": task.query}
        final_output = self.debate_instance.inference(sample)
        self.debate_instance.call_llm = original_call_llm
        print("[LLM_Debate_Wrapper] Original call_llm restored.")

        # 记录完整对话历史
        full_history = []
        agent_contexts = getattr(self.debate_instance, 'agent_contexts', None)
        if agent_contexts is not None:
            for agent_index, context in enumerate(agent_contexts):
                for msg_index, message in enumerate(context):
                    full_history.append({
                        "role": f"Assistant {agent_index+1}" if message["role"] == "assistant" else "User",
                        "role_index": agent_index,
                        "content": message["content"],
                        "msg_index": msg_index,
                        # 可选: 你可以根据 msg_index 或轮次推断 round
                    })

        log = {
            "final_output": final_output,
            "injected_role_index": target_agent_index,
            "injected_role": f"Assistant {target_agent_index + 1}",
            "malicious_action_description": malicious_agent.description,
            "agent_names": [f"Assistant {i+1}" for i in range(agents_num)],
            "full_history": full_history,
        }
        return final_output, log 