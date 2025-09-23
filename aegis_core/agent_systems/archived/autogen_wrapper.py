# file: agent_systems/autogen_wrapper.py

import re
import asyncio  # CHANGED: Import asyncio to run async methods from a sync context
from .base_wrapper import SystemWrapper
from malicious_factory.agent import MaliciousAgent
from typing import Any, Dict, Tuple
from methods import get_method_class

# We assume the user's AutoGen_Main class is in a file we can import
# For this example, let's assume it's at 'methods.autogen.main.AutoGen_Main'
from methods.autogen.autogen_main import AutoGen_Main

class AutoGenWrapper(SystemWrapper):
    """
    A wrapper for the user-provided AutoGen system. It injects malicious
    behavior by monkey-patching the central `call_llm` method.
    This version has been converted to be fully synchronous to avoid issues
    with calling environments that do not support async/await.
    """
    def __init__(self, general_config: Dict[str, Any], method_config: Dict[str, Any]):
        from utils.async_llm import create_llm_instance
        exp_config = general_config.get('experiment_config', {})
        llm_config = exp_config.get('llm_config', {})
        if not llm_config:
            raise ValueError("No LLM configuration found for the specified model name.")
        
        # NOTE: self.llm is still an async-first client, but we will call it synchronously.
        self.llm = create_llm_instance(llm_config)
        method_name = exp_config['system_under_test']['name']  # "autogen"
        dataset_name = exp_config.get('benchmark_name', None)
        MAS_CLASS = get_method_class(method_name, dataset_name)
        self.autogen_instance = MAS_CLASS(general_config, method_config_name=None)
        
        # Override the instance's LLM call with our own for consistency
        # This allows us to use our standardized LLM clients.
        # The actual method being assigned here is what will be patched later.
        # self.autogen_instance.call_llm = self.llm.acall
        print(f"AutoGenWrapper initialized with {MAS_CLASS.__name__}.")

    def run_with_injection(  # CHANGED: Removed 'async' keyword
        self,
        task: Any,
        malicious_agent: MaliciousAgent,
        injection_target: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:

        target_role = injection_target['role']
        # target_instance_id = injection_target.get('instance_id', 0)

        # Store the original call_llm method (which is self.llm.acall)
        original_llm_call = self.autogen_instance.call_llm
        
        call_counters = {"assistant_agent": 0, "user_proxy_agent": 0}

        from malicious_factory import MaliciousAgentFactory
        factory = MaliciousAgentFactory(llm=self.llm)

        from methods.autogen.prompt import ASSISTANT_AGENT_SYSTEM_MESSAGE, ASSISTANT_AGENT_SYSTEM_MESSAGE_CODER, DEFAULT_USER_PROXY_AGENT_SYSTEM_MESSAGE

        # --- Define the new, sophisticated __call__ method for the LLM ---
        # CHANGED: This is now a regular 'def', not 'async def'
        def sophisticated_malicious_llm_call(*args, **kwargs):
    
            # 1. Determine the current speaker (This logic remains correct)
            messages = args[2]
            system_message = messages[0]['content']
            speaker_role = "unknown"
            if system_message in [ASSISTANT_AGENT_SYSTEM_MESSAGE, ASSISTANT_AGENT_SYSTEM_MESSAGE_CODER]:
                speaker_role = "assistant_agent"
            elif system_message == DEFAULT_USER_PROXY_AGENT_SYSTEM_MESSAGE:
                speaker_role = "user_proxy_agent"
            
            if speaker_role == "unknown":
                # Call original synchronous function
                return original_llm_call(args[0], args[1], messages)

            # 2. Update and check the counter
            call_counters[speaker_role] += 1
            current_call_count = call_counters[speaker_role]
            
            print(f"[Runner Intercept] Turn: '{speaker_role}' speaking (Call #{current_call_count}).")

            # 3. Decide whether to inject malice
            is_malicious_call = speaker_role == target_role

            if not is_malicious_call:
                # Call the original synchronous function
                return original_llm_call(args[0], args[1], messages)

            # 4. --- Execute Malicious Logic ---
            print(f"*** Malicious Agent Activated on '{speaker_role}' (Call #{current_call_count}) ***")
            
            task_input = messages[-1]['content']
            response = factory.inject_malicious_behavior(
                lambda: original_llm_call(args[0], args[1], messages),
                malicious_agent,
                task_input=task_input,
                messages=messages
            )
            
            if isinstance(response, str):
                return {'response': response}
            else:
                return response


        # --- Apply the monkey patch ---
        self.autogen_instance.call_llm = sophisticated_malicious_llm_call
        print(f"[Runner] Monkey-patch applied to `call_llm` method. Target: {target_role}.")

        # --- Execute the patched workflow ---
        sample = {"query": task.query}
        # NOTE: This call is now fully synchronous because the patched method is synchronous.
        final_output = self.autogen_instance.inference(sample)

        # --- Cleanup ---
        self.autogen_instance.call_llm = original_llm_call
        print("[Runner] Original `call_llm` method restored.")
        
        log = {
            "final_output": final_output,
            "full_history": self.autogen_instance.history,
            "injected_role": target_role,
            "malicious_action_description": malicious_agent.description
        }
        return final_output, log