#!/usr/bin/env python3
"""
统一数据格式处理脚本
将不同framework的注入失败数据合并成统一格式的训练集
"""

import os
import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import argparse

sys.path.append(str(Path(__file__).parent.parent))
from analyze_injection_errors import analyze_benchmark_errors, parse_filename


@dataclass
class UnifiedTrainingData:
    """统一的训练数据格式"""
    id: str
    metadata: Dict[str, Any]
    input: Dict[str, Any]
    output: Dict[str, Any] 
    ground_truth: Dict[str, Any]


class FrameworkDataProcessor:
    """不同框架数据处理器的基类"""
    
    def __init__(self, schema_path: str):
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        self.agent_name_map = self.schema["agent_name_standardization"]
        self.error_type_map = self.schema["error_type_mapping"]["detailed_mapping"]
    
    def standardize_agent_name(self, original_name: str, framework: str) -> str:
        """标准化智能体名称"""
        name = original_name.strip().lower()
        
        if framework == "agentverse":
            if "roleassigner" in name:
                return "RoleAssigner"
            elif "solver" in name:
                return "Solver"
            elif "critic" in name:
                return "Critic"
            elif "evaluator" in name:
                return "Evaluator"
                
        elif framework == "dylan":
            if "assistant" in name:
                return "Assistant"
            elif "tester" in name:
                return "Tester"
            elif "reflector" in name:
                return "Reflector"
            elif "programmer" in name:
                return "Programmer"
            elif "debugger" in name:
                return "Debugger"
            elif "computerscientist" in name:
                return "ComputerScientist"
            elif "algorithmdeveloper" in name:
                return "AlgorithmDeveloper"
            elif "pythonassistant" in name:
                return "PythonAssistant"
            elif "qualitymanager" in name:
                return "QualityManager"
                
        elif framework == "llm_debate":
            if "assistant" in name:
                import re
                match = re.search(r'(\d+)', name)
                if match:
                    return f"Assistant {match.group(1)}"
                return "Assistant"
            elif "aggregator" in name:
                return "Aggregator"
            else:
                return "Assistant"
                
        elif framework == "macnet":
            if "node" in name:
                import re
                match = re.search(r'node-?(\d+)', name)
                if match:
                    node_num = match.group(1)
                    if '-' in name:
                        return f"Node-{node_num}"
                    else:
                        return f"Node{node_num}"
                return "Node"
        
        return original_name.title()
    
    def standardize_phase(self, original_phase: str) -> str:
        """标准化对话阶段"""
        phase = original_phase.lower()
        
        phase_map = {
            "role_assignment": "initialization",
            "solving": "reasoning", 
            "criticism": "evaluation",
            "evaluation": "evaluation",
            "discussion": "discussion",
            "decision": "decision"
        }
        
        return phase_map.get(phase, "other")
    
    def generate_id(self, metadata: Dict[str, Any], line_number: int) -> str:
        """生成唯一标识符"""
        source_str = f"{metadata['benchmark']}_{metadata['model']}_{metadata['framework']}_{line_number}"
        hash_obj = hashlib.md5(source_str.encode())
        return f"{source_str}_{hash_obj.hexdigest()[:8]}"
    
    def process_sample(self, sample: Dict[str, Any], file_params: Dict[str, Any], line_number: int) -> UnifiedTrainingData:
        """处理单个样本，需要在子类中实现具体逻辑"""
        raise NotImplementedError
    
    def extract_final_output(self, injection_log: Dict[str, Any]) -> str:
        """提取系统的最终输出"""
        final_output = injection_log.get("final_output", {})
        if isinstance(final_output, dict):
            response = final_output.get("response", "")
            if response:
                return response
        elif isinstance(final_output, str):
            return final_output
        
        if "response" in injection_log:
            return injection_log["response"]
        
        return "No final output available"


class AgentVerseProcessor(FrameworkDataProcessor):
    """AgentVerse框架数据处理器"""
    
    def parse_multi_agent_info(self, file_params: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """解析多智能体文件名信息
        Returns: (agent_names, error_types, injection_strategies)
        """
        agent_type = file_params.get("agent_type", "")
        error_type = file_params.get("error_type", "")
        injection_type = file_params.get("injection_type", "")
        
        agent_names = []
        if agent_type:
            parts = agent_type.split("-")
            for part in parts:
                if part:
                    import re
                    role_match = re.match(r'([a-zA-Z]+)\d*', part)
                    if role_match:
                        agent_names.append(role_match.group(1))
        
        error_types = []
        if error_type:
            import re
            matches = re.findall(r'FM-\d+\.\d+', error_type)
            error_types = matches if matches else [error_type]
        
        injection_strategies = []
        if injection_type:
            import re
            cleaned_type = re.sub(r'_?n\d+$', '', injection_type)
            parts = cleaned_type.split("-")
            injection_strategies = [part for part in parts if part]
        
        return agent_names, error_types, injection_strategies
    
    def process_sample(self, sample: Dict[str, Any], file_params: Dict[str, Any], line_number: int) -> UnifiedTrainingData:
        injection_log = sample.get("injection_log", {})
        
        agent_names, error_types, injection_strategies = self.parse_multi_agent_info(file_params)
        
        metadata = {
            "framework": "agentverse",
            "benchmark": file_params.get("benchmark", ""),
            "model": file_params.get("model", ""),
            "num_agents": len(injection_log.get("role_descriptions", [])),
            "num_injected_agents": len(agent_names) or len(injection_log.get("multi_injection_info", [])),
            "task_type": self._infer_task_type(sample.get("tag", []))
        }
        
        conversation_history = []
        history = injection_log.get("conversation_history", [])
        
        for i, entry in enumerate(history):
            std_entry = {
                "step": i + 1,
                "agent_name": self.standardize_agent_name(entry.get("role", ""), "agentverse"),
                "agent_role": entry.get("role", ""),
                "content": entry.get("response", ""),
                "phase": self.standardize_phase(entry.get("phase", "other"))
            }
            conversation_history.append(std_entry)
        
        input_data = {
            "query": sample.get("query", ""),
            "conversation_history": conversation_history,
            "final_output": self.extract_final_output(injection_log)
        }
        
        faulty_agents = []
        multi_injection_info = injection_log.get("multi_injection_info", [])
        
        if multi_injection_info:
            for i, inj_info in enumerate(multi_injection_info):
                error_type = inj_info.get("fm_error_type", "")
                injection_strategy = inj_info.get("injection_strategy", "")
                
                if not error_type and i < len(error_types):
                    error_type = error_types[i]
                if not injection_strategy and i < len(injection_strategies):
                    injection_strategy = injection_strategies[i]
                
                faulty_agent = {
                    "agent_name": self.standardize_agent_name(inj_info.get("injected_role", ""), "agentverse"),
                    "error_type": error_type,
                    "injection_strategy": injection_strategy
                }
                faulty_agents.append(faulty_agent)
        else:
            length = max(len(agent_names), len(error_types), len(injection_strategies), 1)
            for i in range(length):
                agent_name = agent_names[i] if i < len(agent_names) else "Unknown"
                error_type = error_types[i] if i < len(error_types) else ""
                injection_strategy = injection_strategies[i] if i < len(injection_strategies) else ""
                
                faulty_agent = {
                    "agent_name": self.standardize_agent_name(agent_name, "agentverse"),
                    "error_type": error_type,
                    "injection_strategy": injection_strategy
                }
                faulty_agents.append(faulty_agent)
        
        output_data = {
            "faulty_agents": faulty_agents
        }
        
        ground_truth = {
            "correct_answer": sample.get("gt", ""),
            "injected_agents": [
                {
                    "agent_name": fa["agent_name"],
                    "error_type": fa["error_type"],
                    "injection_strategy": fa["injection_strategy"],
                    "malicious_action_description": ""
                } for fa in faulty_agents
            ],
            "is_injection_successful": sample.get("status") == "success"
        }
        
        return UnifiedTrainingData(
            id=self.generate_id(metadata, line_number),
            metadata=metadata,
            input=input_data,
            output=output_data,
            ground_truth=ground_truth
        )
    
    def _infer_task_type(self, tags: List[str]) -> str:
        """推断任务类型"""
        tags_str = " ".join(tags).lower()
        if "math" in tags_str:
            return "math"
        elif "code" in tags_str or "humaneval" in tags_str:
            return "code_generation"
        else:
            return "reasoning"


class DylanProcessor(FrameworkDataProcessor):
    """Dylan框架数据处理器"""
    
    def parse_multi_agent_info(self, file_params: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """解析多智能体文件名信息
        Returns: (agent_names, error_types, injection_strategies)
        """
        agent_type = file_params.get("agent_type", "")
        error_type = file_params.get("error_type", "")
        injection_type = file_params.get("injection_type", "")
        
        agent_names = []
        if agent_type:
            parts = agent_type.split("-")
            for part in parts:
                if part:
                    agent_names.append(part)
        
        error_types = []
        if error_type:
            import re
            matches = re.findall(r'FM-\d+\.\d+', error_type)
            error_types = matches if matches else [error_type]
        
        injection_strategies = []
        if injection_type:
            import re
            cleaned_type = re.sub(r'_?n\d+$', '', injection_type)
            parts = cleaned_type.split("-")
            injection_strategies = [part for part in parts if part]
        
        return agent_names, error_types, injection_strategies
    
    def process_sample(self, sample: Dict[str, Any], file_params: Dict[str, Any], line_number: int) -> UnifiedTrainingData:
        injection_log = sample.get("injection_log", {})
        
        agent_names, error_types, injection_strategies = self.parse_multi_agent_info(file_params)
        
        metadata = {
            "framework": "dylan",
            "benchmark": file_params.get("benchmark", ""),
            "model": file_params.get("model", ""),
            "num_agents": len(set(entry.get("role", "") for entry in injection_log.get("full_history", []))),
            "num_injected_agents": len(agent_names) or 1,
            "task_type": self._infer_task_type(sample.get("tag", []))
        }
        
        conversation_history = []
        history = injection_log.get("conversation_history", [])
        if not history:
            history = injection_log.get("full_history", [])
        
        for i, entry in enumerate(history):
            agent_id = entry.get("agent_id", "")
            role = entry.get("role", "")
            role_index = entry.get("role_index", "")
            if role_index != "":
                role_index = int(role_index) + 1
            content = entry.get("content", entry.get("response", ""))
            
            if agent_id:
                std_agent_name = agent_id.replace(" ", "")
            elif role:
                std_agent_name = role.replace(" ", "")
            else:
                std_agent_name = f"Assistant{i+1}"
            
            if role == "Assistant" and role_index:
                std_agent_name = f"Assistant{role_index}"
            
            std_entry = {
                "step": i + 1,
                "agent_name": std_agent_name,
                "agent_role": std_agent_name,
                "content": content,
                "phase": "reasoning"
            }
            conversation_history.append(std_entry)
        
        input_data = {
            "query": sample.get("query", ""),
            "conversation_history": conversation_history
        }
        
        faulty_agents = []
        if agent_names:
            length = max(len(agent_names), len(error_types), len(injection_strategies), 1)
            for i in range(length):
                agent_name = agent_names[i] if i < len(agent_names) else "Unknown"
                error_type = error_types[i] if i < len(error_types) else ""
                injection_strategy = injection_strategies[i] if i < len(injection_strategies) else ""
                
                if file_params.get("benchmark", "").lower() == "humaneval":
                    import re
                    agent_name = re.sub(r'\d+$', '', agent_name)
                
                faulty_agent = {
                    "agent_name": agent_name,
                    "error_type": error_type,
                    "injection_strategy": injection_strategy
                }
                faulty_agents.append(faulty_agent)
        else:
            injected_role = injection_log.get("injected_role", "Unknown")
            
            if file_params.get("benchmark", "").lower() == "humaneval":
                import re
                injected_role = re.sub(r'\s*\d+$', '', injected_role)
            
            faulty_agent = {
                "agent_name": injected_role,
                "error_type": injection_log.get("fm_error_type", "") or "FM-2.3",
                "injection_strategy": injection_log.get("injection_strategy", "") or "prompt_injection"
            }
            faulty_agents = [faulty_agent]
        
        output_data = {
            "faulty_agents": faulty_agents
        }
        
        ground_truth = {
            "correct_answer": sample.get("gt", ""),
            "injected_agents": [
                {
                    "agent_name": fa["agent_name"],
                    "error_type": fa["error_type"],
                    "injection_strategy": fa["injection_strategy"],
                    "malicious_action_description": injection_log.get("malicious_action_description", "")
                } for fa in faulty_agents
            ],
            "is_injection_successful": sample.get("status") == "success"
        }
        
        return UnifiedTrainingData(
            id=self.generate_id(metadata, line_number),
            metadata=metadata,
            input=input_data,
            output=output_data,
            ground_truth=ground_truth
        )
    
    def _infer_task_type(self, tags: List[str]) -> str:
        tags_str = " ".join(tags).lower()
        if "math" in tags_str:
            return "math"
        elif "code" in tags_str or "humaneval" in tags_str:
            return "code_generation"
        else:
            return "reasoning"


class LLMDebateProcessor(FrameworkDataProcessor):
    """LLM Debate框架数据处理器"""
    
    def parse_multi_agent_info(self, file_params: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """解析多智能体文件名信息
        Returns: (agent_names, error_types, injection_strategies)
        """
        agent_type = file_params.get("agent_type", "")
        error_type = file_params.get("error_type", "")
        injection_type = file_params.get("injection_type", "")
        
        agent_names = []
        if agent_type:
            parts = agent_type.split("-")
            agent_names = [part.replace(" ", "").lower() for part in parts]
        
        error_types = []
        if error_type:
            import re
            matches = re.findall(r'FM-\d+\.\d+', error_type)
            error_types = matches if matches else [error_type]
        
        injection_strategies = []
        if injection_type:
            import re
            cleaned_type = re.sub(r'_?n\d+$', '', injection_type)
            parts = cleaned_type.split("-")
            injection_strategies = [part for part in parts if part]
        
        return agent_names, error_types, injection_strategies
    
    def process_sample(self, sample: Dict[str, Any], file_params: Dict[str, Any], line_number: int) -> UnifiedTrainingData:
        injection_log = sample.get("injection_log", {})
        
        agent_names, error_types, injection_strategies = self.parse_multi_agent_info(file_params)
        
        metadata = {
            "framework": "llm_debate",
            "benchmark": file_params.get("benchmark", ""),
            "model": file_params.get("model", ""),
            "num_agents": len(injection_log.get("agent_names", [])),
            "num_injected_agents": len(agent_names) or 1,
            "task_type": self._infer_task_type(sample.get("tag", []))
        }
        
        conversation_history = []
        history = injection_log.get("full_history", [])
        
        for i, entry in enumerate(history):
            role = entry.get("role", "").replace(" ", "")
            content = entry.get("content", "")
            
            std_agent_name = role if role else f"Agent{i+1}"
            std_agent_name = std_agent_name.replace("Agent", "Assistant")
            std_entry = {
                "step": i + 1,
                "agent_name": std_agent_name,
                "agent_role": std_agent_name,
                "content": content,
                "phase": "discussion"
            }
            conversation_history.append(std_entry)
        
        input_data = {
            "query": sample.get("query", ""),
            "conversation_history": conversation_history,
            "final_output": self.extract_final_output(injection_log)
        }
        
        faulty_agents = []
        if agent_names:
            length = max(len(agent_names), len(error_types), len(injection_strategies), 1)
            for i in range(length):
                agent_name = agent_names[i] if i < len(agent_names) else "Unknown"
                error_type = error_types[i] if i < len(error_types) else ""
                injection_strategy = injection_strategies[i] if i < len(injection_strategies) else ""
                
                if self.standardize_agent_name(agent_name, "llm_debate") == "Aggregator":
                    continue
                faulty_agent = {
                    "agent_name": agent_name.replace("agent", "Assistant"),
                    "error_type": error_type,
                    "injection_strategy": injection_strategy
                }
                faulty_agents.append(faulty_agent)
        else:
            error_type = injection_log.get("fm_error_type", "")
            injection_strategy = injection_log.get("injection_strategy", "")
            
            if not error_type:
                error_type = file_params.get("error_type", "")
            if not injection_strategy:
                injection_strategy = file_params.get("injection_type", "")
            
            faulty_agent = {
                "agent_name": injection_log.get("injected_role", "Unknown"),
                "error_type": error_type,
                "injection_strategy": injection_strategy
            }
            faulty_agents = [faulty_agent]
        
        output_data = {
            "faulty_agents": faulty_agents
        }
        
        ground_truth = {
            "correct_answer": sample.get("gt", ""),
            "injected_agents": [
                {
                    "agent_name": fa["agent_name"],
                    "error_type": fa["error_type"],
                    "injection_strategy": fa["injection_strategy"],
                    "malicious_action_description": injection_log.get("malicious_action_description", "")
                } for fa in faulty_agents
            ],
            "is_injection_successful": sample.get("status") == "success"
        }
        
        return UnifiedTrainingData(
            id=self.generate_id(metadata, line_number),
            metadata=metadata,
            input=input_data,
            output=output_data,
            ground_truth=ground_truth
        )
    
    def _infer_task_type(self, tags: List[str]) -> str:
        tags_str = " ".join(tags).lower()
        if "math" in tags_str:
            return "math"
        elif "code" in tags_str or "humaneval" in tags_str:
            return "code_generation"
        else:
            return "reasoning"


class MacNetProcessor(FrameworkDataProcessor):
    """MacNet框架数据处理器"""
    
    def parse_multi_agent_info(self, file_params: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """解析多智能体文件名信息
        Returns: (agent_names, error_types, injection_strategies)
        """
        agent_type = file_params.get("agent_type", "")
        error_type = file_params.get("error_type", "")
        injection_type = file_params.get("injection_type", "")
        
        agent_names = []
        import re
        matches = re.findall(r'node-?\d+', agent_type)
        agent_names = matches
        
        error_types = []
        if error_type:
            if error_type.count("FM-") > 1:
                import re
                matches = re.findall(r'FM-\d+\.\d+', error_type)
                error_types = matches
            else:
                error_types = [error_type]
        
        injection_strategies = []
        if injection_type:
            import re
            cleaned_type = re.sub(r'_?n\d+$', '', injection_type)
            
            parts = cleaned_type.split("-")
            injection_strategies = [part for part in parts if part and not (part.startswith("n") and part[1:].isdigit())]
        
        return agent_names, error_types, injection_strategies
    
    def process_sample(self, sample: Dict[str, Any], file_params: Dict[str, Any], line_number: int) -> UnifiedTrainingData:
        injection_log = sample.get("injection_log", {})
        
        agent_names, error_types, injection_strategies = self.parse_multi_agent_info(file_params)
        
        metadata = {
            "framework": "macnet",
            "benchmark": file_params.get("benchmark", ""),
            "model": file_params.get("model", ""),
            "num_agents": injection_log.get("total_llm_calls", len(agent_names)),
            "num_injected_agents": len(agent_names),
            "task_type": self._infer_task_type(sample.get("tag", []))
        }
        
        conversation_history = []
        llm_calls = injection_log.get("llm_call_history", [])
        
        for i, call in enumerate(llm_calls):
            if not call.get("injected", False):
                std_entry = {
                    "step": i + 1,
                    "agent_name": self.standardize_agent_name(f"node{call.get('node_id', '')}", "macnet"),
                    "agent_role": call.get("system_prompt", "")[:50],
                    "content": call.get("response", ""),
                    "phase": "reasoning"
                }
                conversation_history.append(std_entry)
        
        input_data = {
            "query": sample.get("query", ""),
            "conversation_history": conversation_history,
            "final_output": self.extract_final_output(injection_log)
        }
        
        faulty_agents = []
        
        for i, agent_name in enumerate(agent_names):
            error_type = error_types[i] if i < len(error_types) else ""
            injection_strategy = injection_strategies[i] if i < len(injection_strategies) else ""
            
            faulty_agent = {
                "agent_name": self.standardize_agent_name(f"node{agent_name}", "macnet"),
                "error_type": error_type,
                "injection_strategy": injection_strategy
            }
            faulty_agents.append(faulty_agent)
        
        output_data = {
            "faulty_agents": faulty_agents
        }
        
        ground_truth = {
            "correct_answer": sample.get("gt", ""),
            "injected_agents": [
                {
                    "agent_name": self.standardize_agent_name(f"node{agent_name}", "macnet"),
                    "error_type": error_types[i] if i < len(error_types) else "",
                    "injection_strategy": injection_strategies[i] if i < len(injection_strategies) else "",
                    "malicious_action_description": f"Node {agent_name} injected with {injection_strategies[i] if i < len(injection_strategies) else 'unknown'}"
                }
                for i, agent_name in enumerate(agent_names)
            ],
            "is_injection_successful": sample.get("status") == "success"
        }
        
        return UnifiedTrainingData(
            id=self.generate_id(metadata, line_number),
            metadata=metadata,
            input=input_data,
            output=output_data,
            ground_truth=ground_truth
        )
    
    def _infer_task_type(self, tags: List[str]) -> str:
        tags_str = " ".join(tags).lower()
        if "math" in tags_str:
            return "math"
        elif "code" in tags_str or "humaneval" in tags_str:
            return "code_generation"
        else:
            return "reasoning"


class NormalSampleProcessor(FrameworkDataProcessor):
    """正常样本处理器 - 处理通过inference.py生成的正样本数据"""
    
    def process_sample(self, sample: Dict[str, Any], file_params: Dict[str, Any], line_number: int) -> UnifiedTrainingData:
        injection_log = sample.get("injection_log", {})
        framework = injection_log.get("framework", file_params.get("framework", "unknown"))
        
        metadata = {
            "framework": framework,
            "benchmark": file_params.get("benchmark", ""),
            "model": file_params.get("model", "gpt-4o-mini"),
            "num_agents": self._estimate_num_agents(injection_log),
            "num_injected_agents": 0,
            "task_type": self._infer_task_type_from_query(sample.get("query", ""))
        }
        
        conversation_history = []
        full_history = injection_log.get("full_history", [])
        
        if not full_history:
            if framework == "macnet":
                std_entry = {
                    "step": 1,
                    "agent_name": "MacNet",
                    "agent_role": "MacNet Agent",
                    "content": injection_log.get("final_output", ""),
                    "phase": "reasoning"
                }
                conversation_history.append(std_entry)
        else:
            for i, entry in enumerate(full_history):
                std_entry = {
                    "step": entry.get("step", i + 1),
                    "agent_name": self.standardize_agent_name(entry.get("agent_name", entry.get("role", "Unknown")), framework),
                    "agent_role": entry.get("role", "Unknown"),
                    "content": entry.get("content", ""),
                    "phase": self._infer_phase_from_entry(entry, framework)
                }
                conversation_history.append(std_entry)
        
        input_data = {
            "query": sample.get("query", ""),
            "conversation_history": conversation_history
        }
        
        output_data = {
            "faulty_agents": []
        }
        
        response = sample.get("response", {})
        if isinstance(response, dict):
            final_response = response.get("response", str(response))
        else:
            final_response = str(response)
        
        ground_truth = {
            "correct_answer": sample.get("gt", final_response),
            "injected_agents": [],
            "is_injection_successful": False,
            "is_normal_sample": True
        }
        
        return UnifiedTrainingData(
            id=self.generate_id(metadata, line_number),
            metadata=metadata,
            input=input_data,
            output=output_data,
            ground_truth=ground_truth
        )
    
    def _estimate_num_agents(self, injection_log: Dict[str, Any]) -> int:
        """估算智能体数量"""
        full_history = injection_log.get("full_history", [])
        if not full_history:
            return 1
        
        agent_names = set()
        for entry in full_history:
            agent_name = entry.get("agent_name", entry.get("role", "Unknown"))
            if agent_name and agent_name != "Unknown":
                agent_names.add(agent_name)
        
        return max(len(agent_names), 1)
    
    def _infer_task_type_from_query(self, query: str) -> str:
        """从查询推断任务类型"""
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["math", "calculate", "solve", "equation", "number"]):
            return "math"
        elif any(keyword in query_lower for keyword in ["code", "program", "function", "implement", "algorithm"]):
            return "code_generation"
        else:
            return "reasoning"
    
    def _infer_phase_from_entry(self, entry: Dict[str, Any], framework: str) -> str:
        """从条目推断对话阶段"""
        if framework == "llm_debate":
            return "discussion"
        elif framework in ["dylan", "dylan_math", "dylan_humaneval", "dylan_mmlu"]:
            return "reasoning"
        elif framework in ["agentverse", "agentverse_humaneval", "agentverse_mgsm"]:
            role = entry.get("role", "").lower()
            if "role" in role or "assign" in role:
                return "initialization"
            elif "solver" in role:
                return "reasoning"
            elif "critic" in role or "eval" in role:
                return "evaluation"
            else:
                return "reasoning"
        elif framework in ["macnet", "macnet_srdd"]:
            return "reasoning"
        else:
            return "other"


class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(self, schema_path: str, input_dir: str, output_dir: str, normal_samples_dir: str = None, separate_datasets: bool = False, only_normal: bool = False, only_negative: bool = False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.normal_samples_dir = normal_samples_dir
        self.separate_datasets = separate_datasets
        self.only_normal = only_normal
        self.only_negative = only_negative
        
        self.processors = {
            "agentverse": AgentVerseProcessor(schema_path),
            "dylan": DylanProcessor(schema_path),
            "llm_debate": LLMDebateProcessor(schema_path),
            "macnet": MacNetProcessor(schema_path)
        }
        
        self.normal_processor = NormalSampleProcessor(schema_path)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def get_framework_from_filename(self, filename: str) -> str:
        """从文件名推断框架类型"""
        if filename.startswith("agentverse"):
            return "agentverse"
        elif filename.startswith("dylan"):
            return "dylan" 
        elif filename.startswith("llm_debate"):
            return "llm_debate"
        elif filename.startswith("macnet"):
            return "macnet"
        else:
            return "unknown"
    
    def build_dataset(self):
        """构建统一格式的数据集"""
        print("🚀 开始构建统一格式的训练数据集...")
        
        all_benchmark_errors = {}
        
        if not self.only_normal:
            print("📊 分析注入实验错误数据...")
            
            benchmark_dirs = [d for d in os.listdir(self.input_dir) if
                                os.path.isdir(os.path.join(self.input_dir, d)) and d !=
                                "smoagents_logs"]
            
            for benchmark in benchmark_dirs:
                benchmark_path = os.path.join(self.input_dir, benchmark)
                print(f"  处理benchmark: {benchmark}")
                
                benchmark_result = analyze_benchmark_errors(benchmark_path, benchmark)
                all_benchmark_errors[benchmark] = benchmark_result
        else:
            print("⏭️ 跳过注入错误数据分析（--only_normal 模式）")
        
        negative_samples = []
        total_processed = 0
        
        if not self.only_normal:
            for benchmark, result in all_benchmark_errors.items():
                answer_errors = result['answer_errors']
                print(f"  📝 {benchmark}: 找到 {len(answer_errors)} 个答题错误样本")
                
                for error_sample in answer_errors:
                    try:
                        file_params = error_sample.get('file_params', {})
                        framework = file_params.get('framework', '')
                        
                        if framework in self.processors:
                            processor = self.processors[framework]
                            
                            unified_sample = processor.process_sample(
                                error_sample, 
                                file_params, 
                                error_sample.get('line_number', 0)
                            )
                            
                            negative_samples.append(unified_sample)
                            total_processed += 1
                            
                            if total_processed % 100 == 0:
                                print(f"    已处理 {total_processed} 个负样本...")
                                
                    except Exception as e:
                        print(f"    ⚠️ 处理样本时出错: {e}")
                        continue
            
            print(f"✅ 负样本处理完成！总共处理了 {total_processed} 个负样本")
        else:
            print("⏭️ 跳过负样本处理（--only_normal 模式）")
        
        positive_samples = []
        if not self.only_negative and self.normal_samples_dir and os.path.exists(self.normal_samples_dir):
            print(f"\n📋 处理正样本数据...")
            normal_count = self._process_normal_samples(positive_samples)
            print(f"✅ 正样本处理完成！总共处理了 {normal_count} 个正样本")
        elif self.only_negative:
            print("⏭️ 跳过正样本处理（--only_negative 模式）")
        elif not self.normal_samples_dir:
            print("⚠️ 未提供正样本目录，跳过正样本处理")
        
        if self.separate_datasets:
            print(f"🎯 分别保存正负样本数据集...")
            if negative_samples:
                print(f"  💾 保存负样本数据集 ({len(negative_samples)} 个样本)")
                self.save_dataset(negative_samples, suffix="negative")
            if positive_samples:
                print(f"  💾 保存正样本数据集 ({len(positive_samples)} 个样本)")  
                self.save_dataset(positive_samples, suffix="positive")
        else:
            unified_data = negative_samples + positive_samples
            print(f"🎯 数据集构建完成！总计 {len(unified_data)} 个样本")
            self.save_dataset(unified_data)
    
    def _process_normal_samples(self, positive_samples: List[UnifiedTrainingData]) -> int:
        """处理正样本数据"""
        normal_count = 0
        
        for root, dirs, files in os.walk(self.normal_samples_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    
                    rel_path = os.path.relpath(file_path, self.normal_samples_dir)
                    path_parts = rel_path.split(os.sep)
                    
                    if len(path_parts) >= 3:
                        benchmark = path_parts[0]
                        model = path_parts[1] 
                        framework_part = path_parts[2]
                        
                        framework = self.get_framework_from_filename(file)
                        if framework == "unknown":
                            framework = framework_part
                        
                        file_params = {
                            "framework": framework,
                            "benchmark": benchmark,
                            "model": model,
                            "filename": file
                        }
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                for line_num, line in enumerate(f, 1):
                                    if line.strip():
                                        try:
                                            sample = json.loads(line)
                                            
                                            if (sample.get("status") == "success" and 
                                                "injection_log" in sample):
                                                
                                                unified_sample = self.normal_processor.process_sample(
                                                    sample, file_params, line_num
                                                )
                                                positive_samples.append(unified_sample)
                                                normal_count += 1
                                                
                                                if normal_count % 100 == 0:
                                                    print(f"    已处理 {normal_count} 个正样本...")
                                        
                                        except json.JSONDecodeError:
                                            continue
                        
                        except Exception as e:
                            print(f"    ⚠️ 处理文件 {file_path} 时出错: {e}")
                            continue
        
        return normal_count
    
    def save_dataset(self, unified_data: List[UnifiedTrainingData], suffix: str = ""):
        """保存数据集到不同格式"""
        print("💾 保存数据集...")
        
        data_dicts = []
        for item in unified_data:
            data_dict = {
                "id": item.id,
                "metadata": item.metadata,
                "input": item.input,
                "output": item.output,
                "ground_truth": item.ground_truth
            }
            data_dicts.append(data_dict)
        
        if suffix:
            base_name = f"unified_training_dataset_{suffix}"
        else:
            base_name = "unified_training_dataset_easy"
        
        jsonl_path = os.path.join(self.output_dir, f"{base_name}.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for data_dict in data_dicts:
                f.write(json.dumps(data_dict, ensure_ascii=False) + '\n')
        
        json_path = os.path.join(self.output_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_dicts, f, ensure_ascii=False, indent=2)
        
        stats_path = self.generate_statistics(data_dicts, suffix)
        
        print(f"✅ 数据集已保存到:")
        print(f"  📄 JSONL格式: {jsonl_path}")
        print(f"  📄 JSON格式: {json_path}")
        if stats_path:
            print(f"  📊 统计报告: {stats_path}")
    
    def generate_statistics(self, data_dicts: List[Dict[str, Any]], suffix: str = ""):
        """生成数据集统计报告"""
        print("📊 生成数据集统计报告...")
        
        stats = {
            "总样本数": len(data_dicts),
            "按框架分布": {},
            "按数据集分布": {},
            "按任务类型分布": {},
            "按错误类型分布": {},
            "按注入策略分布": {},
            "多智能体注入分布": {}
        }
        
        for data in data_dicts:
            metadata = data["metadata"]
            output = data["output"]
            
            framework = metadata["framework"]
            stats["按框架分布"][framework] = stats["按框架分布"].get(framework, 0) + 1
            
            benchmark = metadata["benchmark"]
            stats["按数据集分布"][benchmark] = stats["按数据集分布"].get(benchmark, 0) + 1
            
            task_type = metadata["task_type"]
            stats["按任务类型分布"][task_type] = stats["按任务类型分布"].get(task_type, 0) + 1
            
            for agent in output["faulty_agents"]:
                error_type = agent["error_type"]
                stats["按错误类型分布"][error_type] = stats["按错误类型分布"].get(error_type, 0) + 1
                
                strategy = agent["injection_strategy"]
                stats["按注入策略分布"][strategy] = stats["按注入策略分布"].get(strategy, 0) + 1
            
            num_injected = metadata["num_injected_agents"]
            stats["多智能体注入分布"][str(num_injected)] = stats["多智能体注入分布"].get(str(num_injected), 0) + 1
        
        if suffix:
            stats_filename = f"dataset_statistics_{suffix}.json"
        else:
            stats_filename = "dataset_statistics_easy.json"
        
        stats_path = os.path.join(self.output_dir, stats_filename)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print("\n📈 数据集统计摘要:")
        print(f"  总样本数: {stats['总样本数']}")
        print(f"  框架分布: {stats['按框架分布']}")
        print(f"  数据集分布: {stats['按数据集分布']}")
        print(f"  任务类型分布: {stats['按任务类型分布']}")
        if stats['按错误类型分布']:
            print(f"  错误类型分布 (前5): {dict(list(sorted(stats['按错误类型分布'].items(), key=lambda x: x[1], reverse=True))[:5])}")
        
        return stats_path


def main():
    parser = argparse.ArgumentParser(description="构建统一格式的训练数据集")
    parser.add_argument("--input_dir", default="results_inj", help="输入目录（包含注入实验结果）")
    parser.add_argument("--output_dir", default="data_processing/unified_dataset", help="输出目录")
    parser.add_argument("--schema_path", default="data_processing/unified_schema.json", help="统一格式schema文件路径")
    parser.add_argument("--normal_samples_dir", default=None, help="正样本数据目录（包含通过inference.py生成的正常推理结果）")
    parser.add_argument("--separate", action="store_true", help="分别保存正样本和负样本到不同文件")
    parser.add_argument("--only_normal", action="store_true", help="只处理正样本，跳过负样本处理")
    parser.add_argument("--only_negative", action="store_true", help="只处理负样本，跳过正样本处理")
    
    args = parser.parse_args()
    
    args.normal_samples_dir = "results_right"
    # args.only_normal = True
    args.separate = True
    builder = DatasetBuilder(args.schema_path, args.input_dir, args.output_dir, args.normal_samples_dir, args.separate, args.only_normal, args.only_negative)
    builder.build_dataset()
    
    print("🎉 统一格式训练数据集构建完成！")


if __name__ == "__main__":
    main()