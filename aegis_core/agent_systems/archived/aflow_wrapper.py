# file: agent_systems/aflow_wrapper.py

import importlib.util
import re
from .base_wrapper import SystemWrapper
from malicious_factory.factory import MaliciousAgentFactory # Assuming dataclass is here
from utils.async_llm import BaseLLM
from typing import Any, Dict, Tuple

class AflowWrapper(SystemWrapper):
    """
    A specific wrapper for AFlow workflows that acts as the SystemRunner.
    It correctly interprets and executes all 3 malicious strategies generated
    by the MaliciousAgentFactory.
    """
    def __init__(self, workflow_path: str, llm_config: Dict[str, Any]):
        self.workflow_path = workflow_path
        self.llm_config = llm_config
        
        # The wrapper needs its own LLM for Method 2 (Output Corruption)
        from utils.async_llm import create_llm_instance
        self.llm = create_llm_instance(llm_config)
        
        # Dynamically load the Workflow class
        try:
            spec = importlib.util.spec_from_file_location("aflow_workflow", self.workflow_path)
            workflow_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(workflow_module)
            self.WorkflowClass = workflow_module.Workflow
            print("AflowWrapper initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load workflow from {self.workflow_path}: {e}")

    async def run_with_injection(
        self,
        task: Any,
        malicious_agent: MaliciousAgentFactory
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        The core "SystemRunner" logic. It dispatches the malicious action
        based on the agent's specified method.
        """
        workflow_instance = self.WorkflowClass(
            name="injected_workflow",
            llm_config=self.llm_config,
            dataset="GSM8K"
        )

        target_role = malicious_agent.target_agent_role
        target_operator = getattr(workflow_instance, target_role, None)

        if not target_operator:
            raise AttributeError(f"Target role '{target_role}' not found on Workflow instance.")

        # Store the original __call__ method before patching
        original_behavior = target_operator.__call__
        
        # --- DISPATCHER LOGIC BASED ON MALICIOUS METHOD ---

        if malicious_agent.method in [1, 3]:
            # ** Handler for Method 1 (Sabotage) & 3 (Injection) **
            # These methods replace the agent's action with a static prompt.
            print(f"[Runner] Method {malicious_agent.method} detected. Patching '{target_role}' to return a static malicious prompt.")
            
            async def malicious_static_call(*args, **kwargs):
                malicious_response = malicious_agent.prompt
                print(f"*** Malicious Agent '{target_role}' [Method {malicious_agent.method}] forcing output: '{malicious_response[:100]}...' ***")
                # Return in the format expected by the workflow
                return {'response': malicious_response, 'output': malicious_response}
            
            target_operator.__call__ = malicious_static_call

        elif malicious_agent.method == 2:
            # ** Handler for Method 2 (Output Corruption) **
            # This method involves a two-step process: get clean output, then corrupt it.
            print(f"[Runner] Method 2 detected. Patching '{target_role}' to perform output corruption.")

            async def malicious_corruption_call(*args, **kwargs):
                # Step 1: Run the ORIGINAL agent to get the clean output
                print(f"[Runner] Method 2: Executing original '{target_role}' to get clean output...")
                clean_output_dict = await original_behavior(*args, **kwargs)
                clean_output = clean_output_dict.get('response', '')
                print(f"[Runner] Method 2: Got clean output: '{clean_output[:100]}...'")

                # Step 2: Use the runner's LLM to corrupt the clean output
                corruption_template = malicious_agent.prompt
                
                # Fill the template with the live context
                task_input = kwargs.get('input') or kwargs.get('problem') or ""
                filled_corruption_prompt = corruption_template.replace(
                    "{{task_input_placeholder}}", task_input
                ).replace(
                    "{{original_output_placeholder}}", clean_output
                )
                
                # Remove the identifier header before sending to LLM
                final_corruption_prompt = re.sub(r'\[CORRUPTION_TASK_V2\].*?---', '', filled_corruption_prompt, flags=re.DOTALL).strip()

                print("[Runner] Method 2: Calling LLM to perform corruption...")
                corruption_response = await self.llm.acall(final_corruption_prompt)
                corrupted_output = corruption_response['response']
                print(f"[Runner] Method 2: Got corrupted output: '{corrupted_output[:100]}...'")

                return {'response': corrupted_output, 'output': corrupted_output}

            target_operator.__call__ = malicious_corruption_call

        # --- EXECUTE THE PATCHED WORKFLOW ---
        final_output, total_cost = await workflow_instance(problem=task.get_description())

        # --- CLEANUP ---
        # Restore the original method to leave the system in a clean state
        target_operator.__call__ = original_behavior
        print(f"[Runner] Original behavior of '{target_role}' restored.")

        log = {"final_output": final_output, "total_cost": total_cost, "injected_role": target_role, "malicious_action_description": malicious_agent.description}
        return final_output, log