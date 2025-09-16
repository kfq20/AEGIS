import os
import re
from typing import List, Dict, Any, Set, Tuple

from methods.mas_base import MAS
from .prompt_humaneval import *

class AgentVerse_HumanEval(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_humaneval" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        
        self.max_turn = self.method_config['max_turn']
        self.cnt_agents = self.method_config['cnt_agents']
        self.max_criticizing_rounds = self.method_config['max_criticizing_rounds']
        
        self.advice = "No advice yet."
        self.history = []
    
    def inference(self, sample):

        query = sample["query"]
        
        for i in range(self.max_turn):
            
            # Assign roles to agents
            role_descriptions = self.assign_roles(query)
            solution = self.group_vertical_solver_first(query, role_descriptions)

            Score, feedback = self.evaluate(query, role_descriptions, solution)

            if Score == 1:
                break
            else:
                self.advice = feedback

        return {"response": solution}

    def assign_roles(self, query: str):
        # Fetch prompts from config.yaml (assumed to be loaded earlier)
        prepend_prompt = ROLE_ASSIGNER_PREPEND_PROMPT.format(query=query, cnt_agents=self.cnt_agents, advice=self.advice)
        append_prompt = ROLE_ASSIGNER_APPEND_PROMPT.format(cnt_agents=self.cnt_agents)
        
        # Call LLM to get the role assignments
        assigner_messages = self.construct_messages(prepend_prompt, [], append_prompt)
        role_assignment_response = self.call_llm(None, None, assigner_messages)

        # Extract role descriptions using regex
        role_descriptions = self.extract_role_descriptions(role_assignment_response)
        return role_descriptions

    def extract_role_descriptions(self, response: str):
        """
        Extracts the role descriptions from the model's response using regex.
        Assumes the response is formatted like:
        1. an electrical engineer specified in the field of xxx.
        2. an economist who is good at xxx.
        ...
        """
        role_pattern = r"\d+\.\s*([^.]+)"  # 修改正则，匹配数字后和句点之间的内容
        
        role_descriptions = re.findall(role_pattern, response)
        
        if len(role_descriptions) == self.cnt_agents:
            return role_descriptions
        else:
            raise ValueError(f"wrong cnt_agent, expect {self.cnt_agents} agents while we find {len(role_descriptions)} role_descriptions.")

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
        solver_prepend_prompt = SOLVER_PREPEND_PROMPT.format(query=query)
        solver_append_prompt = SOLVER_APPEND_PROMPT.format(role_description=role_descriptions[0])

        solver_message = self.construct_messages(solver_prepend_prompt, history_solver, solver_append_prompt)
        solver_response = self.call_llm(None, None, solver_message)
        self.history.append(
            {
                "role": "assistant",
                "content": f"[{role_descriptions[0]}]: {self.parse_solver(solver_response)}",
            }
        )
        if len(self.history) > max_history_critic:
            history_critic = self.history[-max_history_critic:]
        else:
            history_critic = self.history
        previous_plan = self.parse_solver(solver_response)  # Set the solution as previous_plan
        
        cnt_critic_agent = self.cnt_agents - 1
        
        for i in range(self.max_criticizing_rounds):
            
            #step 2: Critics review the solution
            reviews = []
            for j in range(cnt_critic_agent):
                critic_prepend_prompt = CRITIC_PREPEND_PROMPT.format(query=query)
                critic_append_prompt = CRITIC_APPEND_PROMPT.format(role_description=role_descriptions[j+1])

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
                break
            solver_message = self.construct_messages(solver_prepend_prompt, history_solver, solver_append_prompt)
            solver_response = self.call_llm(None, None, solver_message)
            self.history.append(
                {
                    "role": "assistant",
                    "content": f"[{role_descriptions[0]}]: {self.parse_solver(solver_response)}",
                }
            )
            if len(self.history) > max_history_critic:
                history_critic = self.history[-max_history_critic:]
            else:
                history_critic = self.history
            previous_plan = self.parse_solver(solver_response)
        results = previous_plan
        return results
            
    def evaluate(self, query: str, role_descriptions: List[str], plan):
        evaluator_prepend_prompt = EVALUATOR_PREPEND_PROMPT.format(query=query, all_role_description = "\n".join(role_descriptions), solution = plan)
        evaluator_append_prompt = EVALUATOR_APPEND_PROMPT
        evaluator_message = self.construct_messages(evaluator_prepend_prompt, [], evaluator_append_prompt)
        evaluator_response = self.call_llm(None, None, evaluator_message)
        return self.parse_evaluator(evaluator_response)
        
    def parse_evaluator(self, output) -> Tuple[List[int], str]:

        correctness_match = re.search(r"Score:\s*(\d)", output)
        if correctness_match:
            correctness = int(correctness_match.group(1))
        else:
            raise ValueError("Correctness not found in the output text.")

        advice_match = re.search(r"Response:\s*(.+)", output, re.DOTALL)  
        if advice_match:
            advice = advice_match.group(1).strip()  
            clean_advice = re.sub(r"\n+", "\n", advice.strip())
        else:

            raise ValueError("Advice not found in the output text.")
 
        return correctness, clean_advice
    
    def parse_solver(self, output) -> str:
        return re.findall(r"```.*?\n(.+?)```", output, re.DOTALL)[-1]

    def parse_critic(self, output) -> str:
        if "[Agree]" in output:
            return ""
        else:
            return output

    def construct_messages(self, prepend_prompt: str, history: List[Dict], append_prompt: str):
        messages = []
        if prepend_prompt:
            messages.append({"role": "system", "content": prepend_prompt})
        if len(history) > 0:
            messages += history
        if append_prompt:
            messages.append({"role": "user", "content": append_prompt})
        return messages
