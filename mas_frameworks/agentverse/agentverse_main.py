import os
import re
from typing import List, Dict, Any, Set, Tuple

from methods.mas_base import MAS
from .prompt_main import *

# Define the NEWMAS class which inherits from MAS and implements the inference method
class AgentVerse_Main(MAS):
    def __init__(self, general_config, method_config_name = None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        
        self.max_turn = self.method_config['max_turn']
        self.cnt_agents = self.method_config['cnt_agents']
        self.max_criticizing_rounds = self.method_config['max_criticizing_rounds']
        
        self.dimensions: List[str] = ["Score", "Response"]
        self.advice = "No advice yet."
        self.history = []
    
    def inference(self, sample):

        query = sample["query"]

        for i in range(self.max_turn):
            # Assign roles to agents
            role_descriptions = self.assign_roles(query)

            # Collaborate to solve the query
            solution = self.group_vertical_solver_first(query, role_descriptions)

            # Get evaluation and feedback
            score, feedback = self.evaluate(query, role_descriptions, solution)

            if score == 1:
                break
            else:
                self.advice = feedback
        return {"response": solution}

    def assign_roles(self, query: str):
        # Fetch prompts from config.yaml (assumed to be loaded earlier)
        prepend_prompt = ROLE_ASSIGNER_PREPEND_PROMPT.replace("${query}", query).replace("${cnt_agents}", str(self.cnt_agents)).replace("${advice}", self.advice)
        append_prompt = ROLE_ASSIGNER_APPEND_PROMPT.replace("${cnt_agents}", str(self.cnt_agents))
        
        # Call LLM to get the role assignments
        assigner_messages = self.construct_messages(prepend_prompt, [], append_prompt)
        role_assignment_response = self.call_llm(None, None, assigner_messages)
        # Extract role descriptions using regex
        role_descriptions = self.extract_role_descriptions(role_assignment_response)
        return role_descriptions

    def extract_role_descriptions(self, response: str):
        """
        Extracts the role descriptions from the model's response using regex.
        Handles malicious injection cases with fallback role assignments.
        Assumes the response is formatted like:
        1. an electrical engineer specified in the field of xxx.
        2. an economist who is good at xxx.
        ...
        """
        role_pattern = r"\d+\.\s*([^.]+)"  # extract the content between the number and the period
        
        role_descriptions = re.findall(role_pattern, response)
        
        if len(role_descriptions) == self.cnt_agents:
            return role_descriptions
        else:
            print(f"[AgentVerse] Warning: Expected {self.cnt_agents} agents but found {len(role_descriptions)} role descriptions")
            print(f"[AgentVerse] RoleAssigner output: {response[:300]}...")
            
            # Handle malicious injection by providing fallback roles
            if len(role_descriptions) == 0:
                # Complete parsing failure - use generic roles
                fallback_roles = [f"a general problem solver (agent {i+1})" for i in range(self.cnt_agents)]
                print(f"[AgentVerse] Using fallback roles: {fallback_roles}")
                return fallback_roles
            elif len(role_descriptions) < self.cnt_agents:
                # Partial parsing - pad with generic roles
                while len(role_descriptions) < self.cnt_agents:
                    role_descriptions.append(f"a general problem solver (agent {len(role_descriptions)+1})")
                print(f"[AgentVerse] Padded roles to: {role_descriptions}")
                return role_descriptions
            else:
                # Too many roles - truncate to required number
                truncated_roles = role_descriptions[:self.cnt_agents]
                print(f"[AgentVerse] Truncated roles to: {truncated_roles}")
                return truncated_roles

    def group_vertical_solver_first(self, query: str, role_descriptions: List[str]):
        max_history_solver = 5
        max_history_critic = 3
        previous_plan = "No solution yet."
        # Initialize history and other variables
        nonempty_reviews = []
        history_solver = []
        history_critic = []
        consensus_reached = False
        
        if not self.advice == "No advice yet.":
            self.history.append(
                {
                    "role": "assistant",
                    "content": f"[Evaluator]: {self.advice}",
                }
            )
            if len(self.history) > max_history_solver:
                history_solver = self.history[-max_history_solver:]
            else:
                history_solver = self.history
        # Step 1: Solver generates a solution
        solver_prepend_prompt = SOLVER_PREPEND_PROMPT.replace("${query}", query)
        solver_append_prompt = SOLVER_APPEND_PROMPT.replace("${role_description}", role_descriptions[0])
        # print(f"history_solver: {history_solver}")
        solver_message = self.construct_messages(solver_prepend_prompt, history_solver, solver_append_prompt)
        solver_response = self.call_llm(None, None, solver_message)
        self.history.append(
            {
                "role": "assistant",
                "content": f"[{role_descriptions[0]}]: {solver_response}",
            }
        )
        if len(self.history) > max_history_critic:
            history_critic = self.history[-max_history_critic:]
        else:
            history_critic = self.history
        previous_plan = solver_response  # Set the solution as previous_plan
        
        cnt_critic_agent = self.cnt_agents - 1
        
        for i in range(self.max_criticizing_rounds):
            
            #step 2: Critics review the solution
            reviews = []
            for j in range(cnt_critic_agent):
                critic_prepend_prompt = CRITIC_PREPEND_PROMPT.replace("${query}", query).replace("${role_description}", role_descriptions[j+1])
                critic_append_prompt = CRITIC_APPEND_PROMPT
                critic_message = self.construct_messages(critic_prepend_prompt, history_critic, critic_append_prompt)
                critic_response = self.call_llm(None, None, critic_message)
                if "[Agree]" not in critic_response:
                    self.history.append(
                        {
                            "role": "assistant",
                            "content": f"[{role_descriptions[j+1]}]: {self.parse_critic(critic_response)}",
                        }
                    )
                    if len(self.history) > max_history_solver:
                        history_solver = self.history[-max_history_solver:]
                    else:
                        history_solver = self.history
                reviews.append(critic_response)
            for review in reviews:
                if "[Agree]" not in review:
                    nonempty_reviews.append(review)
            if len(nonempty_reviews) == 0:
                # print("Consensus Reached!")
                break
            solver_message = self.construct_messages(solver_prepend_prompt, history_solver, solver_append_prompt)
            solver_response = self.call_llm(None, None, solver_message)
            self.history.append(
                {
                    "role": "assistant",
                    "content": f"[{role_descriptions[0]}]: {solver_response}",
                }
            )
            if len(self.history) > max_history_critic:
                history_critic = self.history[-max_history_critic:]
            else:
                history_critic = self.history
            previous_plan = solver_response
        results = previous_plan
        return results
    
    def parse_critic(self, output) -> str:
        output = re.sub(r"\n+", "\n", output.strip())
        if "[Agree]" in output:
            return ""
        else:
            return output
            
    def evaluate(self, query: str, role_descriptions: List[str], Plan):
        evaluator_prepend_prompt = EVALUATOR_PREPEND_PROMPT.replace("${query}", query).replace("${all_role_description}", "\n".join(role_descriptions)).replace("${solution}", Plan)
        evaluator_append_prompt = EVALUATOR_APPEND_PROMPT
        evaluator_message = self.construct_messages(evaluator_prepend_prompt, [], evaluator_append_prompt)
        evaluator_response = self.call_llm(None, None, evaluator_message)
        return self.parse_evaluator(evaluator_response)
        
    def parse_evaluator(self, output) -> Tuple[List[int], str]:
        """
        Parse evaluator output with error handling for malicious injection cases.
        Returns (correctness, advice) where correctness defaults to 0 if parsing fails.
        """
        try:
            correctness_match = re.search(r"Correctness:\s*(\d)", output)
            if correctness_match:
                correctness = int(correctness_match.group(1))
            else:
                print(f"[AgentVerse] Warning: Correctness not found in evaluator output, defaulting to 0")
                print(f"[AgentVerse] Evaluator output: {output[:200]}...")
                correctness = 0  # Default to incorrect when parsing fails

            advice_match = re.search(r"Response:\s*(.+)", output, re.DOTALL)  
            if advice_match:
                advice = advice_match.group(1).strip()  
                clean_advice = re.sub(r"\n+", "\n", advice.strip())
            else:
                print(f"[AgentVerse] Warning: Response not found in evaluator output, using full output as advice")
                clean_advice = f"Parsing failed - Raw evaluator output: {output}"
 
            return correctness, clean_advice
            
        except Exception as e:
            print(f"[AgentVerse] Error parsing evaluator output: {e}")
            print(f"[AgentVerse] Raw output: {output}")
            # Return default values to allow execution to continue
            return 0, f"Evaluator parsing error: {str(e)} - Raw output: {output}"

    def construct_messages(self, prepend_prompt: str, history: List[Dict], append_prompt: str):
        messages = []
        if prepend_prompt:
            messages.append({"role": "system", "content": prepend_prompt})
        if len(history) > 0:
            messages += history
        if append_prompt:
            messages.append({"role": "user", "content": append_prompt})
        return messages
