# file: malicious_factory/factory.py

import re
from .base_strategy import Task, BaseLLM
from .strategies.method_1_sabotage import Method1SabotageStrategy
from .strategies.method_2_corruption import Method2CorruptionStrategy
from .strategies.method_3_injection import Method3InjectionStrategy
from .agent import MaliciousAgent # Assuming we define this dataclass elsewhere

class MaliciousAgentFactory:
    def __init__(self, llm: BaseLLM = None):
        self.llm = llm
        # Instantiate all available strategies
        self._strategies = {
            1: Method1SabotageStrategy(),
            2: Method2CorruptionStrategy(),
            3: Method3InjectionStrategy(llm)
        }

    async def create_agent(
        self,
        task_query: Task,
        target_agent_role: str,
        target_agent_index: int,
        method: int,

    ) -> MaliciousAgent:
        
        strategy = self._strategies.get(method)
        if not strategy:
            raise ValueError(f"Invalid malicious method: {method}")

        # Delegate the generation to the chosen strategy object
        prompt, description = await strategy.generate(task_query, target_agent_role, target_agent_index)

        return MaliciousAgent(
            target_agent_role=target_agent_role,
            target_agent_index=target_agent_index,
            method=method,

            prompt=prompt,
            description=description
        )

    def inject_malicious_behavior(self, original_call_func, malicious_agent, **context):
        """
        统一的恶意注入逻辑
        
        Args:
            original_call_func: 原始调用的函数（无参数）
            malicious_agent: 恶意 agent 对象
            **context: 上下文信息（如 task_input, messages 等）
        
        Returns:
            注入后的响应
        """
        if malicious_agent.method == 1:
            return self._method_1_injection(original_call_func, malicious_agent, **context)
        elif malicious_agent.method == 2:
            return self._method_2_injection(original_call_func, malicious_agent, **context)
        elif malicious_agent.method == 3:
            return self._method_3_injection(original_call_func, malicious_agent, **context)
        else:
            raise ValueError(f"Unknown malicious method: {malicious_agent.method}")

    def _method_1_injection(self, original_call_func, malicious_agent, **context):
        """Method 1: Sabotage - 直接返回恶意 prompt"""
        print(f"*** Method 1 (Sabotage) injection activated ***")
        return malicious_agent.prompt

    def _method_2_injection(self, original_call_func, malicious_agent, **context):
        """Method 2: Output Corruption - 获取原始输出后腐蚀"""
        print(f"*** Method 2 (Corruption) injection activated ***")
        
        # 获取原始输出
        clean_output = original_call_func()
        
        # 构建 corruption prompt
        corruption_template = malicious_agent.prompt
        task_input = context.get('task_input', '')
        
        filled_prompt = corruption_template.replace(
            "{task_input_placeholder}", task_input
        ).replace(
            "{original_output_placeholder}", clean_output
        )
        
        # 清理 prompt 并调用 LLM
        final_corruption_prompt = re.sub(
            r'\[CORRUPTION_TASK_V2\].*?---', '', 
            filled_prompt, flags=re.DOTALL
        ).strip()
        
        corrupted_response = self.llm.call(final_corruption_prompt)
        return corrupted_response

    def _method_3_injection(self, original_call_func, malicious_agent, **context):
        """Method 3: Injection - 直接返回恶意 prompt"""
        print(f"*** Method 3 (Injection) injection activated ***")
        return malicious_agent.prompt