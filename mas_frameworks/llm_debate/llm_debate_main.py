# The official implementation of LLM Debate https://github.com/composable-models/llm_multiagent_debate offen encounters errors.
# This is a modified version of the original implementation.

import os
from ..mas_base import MAS

class LLM_Debate_Main(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        self.agents_num = self.method_config["agents_num"]
        self.rounds_num = self.method_config["rounds_num"]
    
    def inference(self, sample):

        query = sample["query"]

        agent_contexts = [[{"role": "user", "content": f"""{query} Make sure to state your answer at the end of the response."""}] for agent in range(self.agents_num)]

        for round in range(self.rounds_num):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = self.construct_message(agent_contexts_other, query, 2*round - 1)
                    agent_context.append(message)

                response = self.call_llm(messages=agent_context)
                agent_context.append({"role": "assistant", "content": response})
        
        answers = [agent_context[-1]['content'] for agent_context in agent_contexts]
        
        final_answer = self.aggregate(query, answers)
        self.agent_contexts = agent_contexts  # 记录完整对话历史，供 wrapper 使用
        return {"response": final_answer}
    
    def construct_message(self, agents, question, idx):

        # Use introspection in the case in which there are no other agents.
        if len(agents) == 0:
            return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

        prefix_string = "These are the recent/updated opinions from other agents: "

        for agent in agents:
            agent_response = agent[idx]["content"]
            response = "\n\n One agent response: ```{}```".format(agent_response)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response. \n The original problem is {}.".format(question)
        return {"role": "user", "content": prefix_string}

    def aggregate(self, query, answers):
        aggregate_instruction = f"Task:\n{query}\n\n"
        for i, answer in enumerate(answers):
            aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
        aggregate_instruction += "Given all the above solutions, reason over them carefully and provide a final answer to the task."
        response = self.call_llm(prompt=aggregate_instruction)
        return response