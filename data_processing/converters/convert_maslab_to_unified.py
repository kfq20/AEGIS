#!/usr/bin/env python3
"""
Convert datasets/MAD_full_dataset.json (MAD dataset) into unified training data format.
- Drop samples where mast_annotation values are all 0 (no mistakes)
- Map trajectory to standardized conversation_history per unified_schema.json
- Since MAD doesn't label faulty agents, leave output.faulty_agents and ground_truth.injected_agents empty
- Mark is_injection_successful True for kept samples (non-zero annotations)
Output: data_processing/unified_dataset/unified_training_dataset_MAD.json
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from pathlib import Path
import sys
import re
import json

def parse_chatdev_log_to_conversation_history(log_string: str) -> dict:
    """
    将一个冗长的 ChatDev 日志字符串解析成结构化的多轮对话历史。

    Args:
        log_string: 原始的日志文件字符串内容。

    Returns:
        一个包含 "conversation_history" 列表的字典。
    """
    conversation_history = []
    step_counter = 1

    # 正则表达式，用于匹配每一次对话的开始标志、角色、阶段和内容。
    # (?s) 标志让 '.' 可以匹配换行符，这对于捕获多行内容至关重要。
    pattern = re.compile(
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO\] "  # 1. 匹配时间戳前缀
        r"([\w\s]+?): "                                  # 2. 捕获 Agent 名称 (例如 "Chief Product Officer")
        r"\*\*.*? on : ([\w]+), turn \d+\*\*"             # 3. 匹配元信息并捕获 Phase 名称 (例如 "DemandAnalysis")
        r"(?s)(.*?)"                                     # 4. 捕获该次对话的全部内容
        r"(?=\n\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO\] |\Z)", # 5. 匹配直到下一个时间戳日志行或字符串结尾
        re.MULTILINE
    )

    # 查找所有匹配的对话轮次
    for match in pattern.finditer(log_string):
        agent_name = match.group(1).strip()
        phase = match.group(2).strip()
        content = match.group(3).strip()

        # 构建并添加当前对话轮次的字典
        conversation_step = {
            "step": step_counter,
            "agent_name": agent_name,
            "agent_role": agent_name,  # 根据格式要求，role 和 name 相同
            "content": content,
            "phase": phase
        }
        conversation_history.append(conversation_step)
        step_counter += 1
        
    return {"conversation_history": conversation_history}

def parse_metagpt_traj_to_conversation_history(log_string: str) -> dict:
    """
    将 MetaGPT 的 trajectory 日志字符串解析成结构化的多轮对话历史。

    Args:
        log_string: 原始的 MetaGPT 日志文件字符串内容。

    Returns:
        一个包含 "conversation_history" 列表的字典。
    """
    conversation_history = []
    step_counter = 1

    # 根据 MetaGPT Agent 的名称推断其所处的开发阶段
    phase_mapping = {
        "Human": "Requirement",
        "SimpleCoder": "Coding",
        "SimpleTester": "Testing",
        "SimpleReviewer": "Reviewing",
        # 可以根据需要添加其他角色到阶段的映射
    }

    # 1. 清理日志，只保留核心通信内容
    log_content_match = re.search(r"===(.*)===", log_string, re.DOTALL)
    if not log_content_match:
        return {"conversation_history": []}
    
    clean_log = log_content_match.group(1).strip()
    
    # 2. 使用长横线作为分隔符，将日志分割成独立的对话块
    blocks = clean_log.split("--------------------------------------------------------------------------------")

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        agent_name = ""
        content = ""

        # 3. 根据块的特征，分别处理 "Human" 的初始需求和其他 Agent 的消息
        # 情况 A: 处理 Human 的初始需求
        if "FROM: Human" in block:
            agent_match = re.search(r"FROM: (\w+)", block)
            content_match = re.search(r"CONTENT:\n(.*)", block, re.DOTALL)
            if agent_match and content_match:
                agent_name = agent_match.group(1).strip()
                content = content_match.group(1).strip()
        
        # 情况 B: 处理其他 Agent 的消息
        else:
            message_part_match = re.search(r"NEW MESSAGES:\n\n(.*)", block, re.DOTALL)
            if message_part_match:
                message_part = message_part_match.group(1).strip()
                # Agent 名称通常是内容的第一行，以冒号结尾
                if ':' in message_part:
                    parts = message_part.split(':', 1)
                    agent_name = parts[0].strip()
                    content = parts[1].strip()

        # 4. 如果成功提取，则构建字典并添加到历史记录中
        if agent_name and content:
            # 使用映射来获取 phase，如果找不到则默认使用 agent_name
            phase = phase_mapping.get(agent_name, agent_name) 
            
            conversation_step = {
                "step": step_counter,
                "agent_name": agent_name,
                "agent_role": agent_name,
                "content": content,
                "phase": phase
            }
            conversation_history.append(conversation_step)
            step_counter += 1

    return {"conversation_history": conversation_history}


def parse_openmanus_traj_to_conversation_history(log_string: str) -> dict:
    """
    Parses an OpenManus trajectory log string into a structured multi-turn 
    conversation history.

    Args:
        log_string: The raw OpenManus log file content as a string.

    Returns:
        A dictionary containing the "conversation_history" list.
    """
    conversation_history = []
    
    # --- Step 1: Extract the initial plan to map step numbers to phases ---
    plan_steps = {}
    plan_match = re.search(r"Plan: .*?\n=+\n(.*?)(?=\n\n\d{4}-)", log_string, re.DOTALL)
    if plan_match:
        plan_block = plan_match.group(1)
        # Find all numbered steps in the plan block
        step_matches = re.findall(r"(\d+)\. \[ \] (.*?)\n", plan_block)
        for num, desc in step_matches:
            plan_steps[int(num)] = desc.strip()

    # --- Step 2: Split the entire log into parts based on "Executing step..." ---
    # The delimiter is the timestamped line indicating a new step is starting.
    delimiter_pattern = r"\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \| INFO\s+\| app\.agent\.base:run:140 - Executing step "
    log_parts = re.split(delimiter_pattern, log_string)

    # The first part is initialization logs, we start from the second part (step 1)
    current_step_num = 1
    for i, part in enumerate(log_parts[1:]):
        # The agent is consistently "Manus"
        agent_name = "Manus"
        
        # Combine thoughts and all tool results into one content block for the step
        full_content = []

        # Find agent's thoughts in the current part
        thought_match = re.search(r"app\.agent\.toolcall:think:81 - ✨ Manus's thoughts: (.*?)(?=\n\d{4}-|\Z)", part, re.DOTALL)
        if thought_match:
            thoughts = thought_match.group(1).strip()
            full_content.append(f"Thoughts:\n{thoughts}")

        # Find all tool action results in the current part
        action_matches = re.findall(r"app\.agent\.toolcall:act:150 - 🎯 Tool.*?Result: (.*?)(?=\n\d{4}-|\Z)", part, re.DOTALL)
        if action_matches:
            actions_summary = "\n\nTool Actions & Results:\n" + "\n---\n".join([res.strip() for res in action_matches])
            full_content.append(actions_summary)
            
        if not full_content:
            continue

        # Use the initial plan to determine the phase for the current step
        # Note: The log shows step numbers can repeat (e.g., after an error), so we look at the step number from the log itself.
        step_header_match = re.match(r"(\d+)/\d+", part)
        if step_header_match:
            log_step_index = int(step_header_match.group(1)) - 1 # Adjust for 0-based index from plan
            phase = plan_steps.get(log_step_index, "Execution")
        else:
            phase = "Execution"

        conversation_step = {
            "step": current_step_num,
            "agent_name": agent_name,
            "agent_role": agent_name,
            "content": "\n\n".join(full_content),
            "phase": phase
        }
        conversation_history.append(conversation_step)
        current_step_num += 1

    return {"conversation_history": conversation_history}

import re
import json

def parse_magentic_traj_to_conversation_history(log_string: str) -> dict:
    """
    Parses a Magentic trajectory log string into a structured multi-turn
    conversation history.

    Args:
        log_string: The raw Magentic log file content as a string.

    Returns:
        A dictionary containing the "conversation_history" list.
    """
    conversation_history = []
    step_counter = 1

    # Phase mapping based on the role of the agent in the log
    phase_mapping = {
        "user": "Requirement",
        "MagenticOneOrchestrator": "Reasoning",
        "WebSurfer": "Execution",
        "Assistant": "Execution",
        "ComputerTerminal": "Execution",
        "FileSurfer": "Execution",
    }

    # 1. Isolate the core conversation from the setup and closing logs
    core_log_match = re.search(r"SCENARIO\.PY STARTING !#!#(.*?)SCENARIO\.PY COMPLETE !#!#", log_string, re.DOTALL)
    if not core_log_match:
        return {"conversation_history": []}
    
    core_log = core_log_match.group(1).strip()

    # 2. Split the log into alternating agent names and content blocks
    # The regex captures the agent names in the delimiter, so the resulting list
    # will be [content_before_first_delimiter, agent1, content1, agent2, content2, ...]
    parts = re.split(r"\n---------- (.*?) ----------\n", core_log)
    
    # The first element is the content before the first delimiter, which is usually empty.
    # We iterate over the remaining parts, taking them in pairs (agent, content).
    it = iter(parts[1:])
    for agent_name in it:
        content = next(it, "").strip()
        
        # Handle the final answer which is part of the orchestrator's last turn
        final_answer_match = re.search(r"FINAL ANSWER: (.*)", content, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            # Remove the final answer part from the main content to avoid duplication
            content = content.replace(final_answer_match.group(0), "").strip()
            
            # Add the main content of the orchestrator if it's not empty
            if content:
                conversation_step = {
                    "step": step_counter,
                    "agent_name": agent_name,
                    "agent_role": agent_name,
                    "content": content,
                    "phase": phase_mapping.get(agent_name, "Reasoning")
                }
                conversation_history.append(conversation_step)
                step_counter += 1

            # Add the final answer as its own step
            conversation_step = {
                "step": step_counter,
                "agent_name": agent_name,
                "agent_role": agent_name,
                "content": final_answer,
                "phase": "Final Answer"
            }
            conversation_history.append(conversation_step)
            step_counter += 1
        
        elif content:
            conversation_step = {
                "step": step_counter,
                "agent_name": agent_name,
                "agent_role": agent_name,
                "content": content,
                "phase": phase_mapping.get(agent_name, "Execution")
            }
            conversation_history.append(conversation_step)
            step_counter += 1

    return {"conversation_history": conversation_history}

def parse_ag2_traj_to_conversation_history(log_string: str) -> dict:
    """
    Parses an ag2 trajectory log string with a YAML-like structure into a 
    standardized conversation history.

    Args:
        log_string: The raw ag2 log file content as a string.

    Returns:
        A dictionary containing the "conversation_history" list.
    """
    conversation_history = []
    step_counter = 1

    # 1. Isolate the core trajectory content from the metadata
    # We are looking for the content that starts right after the 'trajectory:' line
    trajectory_content_match = re.search(r"trajectory:\s*(.*)", log_string, re.DOTALL)
    if not trajectory_content_match:
        return {"conversation_history": []}
        
    trajectory_content = trajectory_content_match.group(1)

    # 2. Define a regex pattern to capture each conversational turn
    # This pattern finds blocks starting with 'content:' and captures the
    # content, role, and name fields.
    pattern = re.compile(
        r"content:\s*(.*?)\s*role:\s*(.*?)\s*name:\s*(.*?)(?=\n\s*content:|\Z)",
        re.DOTALL
    )

    # 3. Find all turns and build the history list
    for match in pattern.finditer(trajectory_content):
        content = match.group(1).strip()
        # role = match.group(2).strip() # Role is available but 'name' is more specific
        agent_name = match.group(3).strip()

        # If content is not empty, create a step
        if content:
            conversation_step = {
                "step": step_counter,
                "agent_name": agent_name,
                "agent_role": agent_name,  # Using the specific name for the role
                "content": content,
                "phase": "Problem Solving" # A generic phase for this format
            }
            conversation_history.append(conversation_step)
            step_counter += 1

    return {"conversation_history": conversation_history}

def parse_appworld_traj_to_conversation_history(log_string: str) -> dict:
    """
    Parses an AppWorld trajectory log string into a structured multi-turn
    conversation history.

    Args:
        log_string: The raw AppWorld log file content as a string.

    Returns:
        A dictionary containing the "conversation_history" list.
    """
    conversation_history = []
    step_counter = 1

    # Define regex patterns for different log entries
    patterns = {
        'Supervisor': re.compile(r"Response from Supervisor Agent\n(.*?)(?=\n\n\n|Code Execution Output)", re.DOTALL),
        'spotify': re.compile(r"Response from spotify Agent\n(.*?)(?=\n\n\n|Message to spotify Agent|Reply from spotify Agent)", re.DOTALL),
        'Execution': re.compile(r"Code Execution Output\n\n(.*?)(?=\n\n\n\n|Entering spotify Agent message loop|Message to Supervisor Agent)", re.DOTALL),
    }

    # Split the log by the main interaction marker to process each turn
    turns = log_string.split("Response from Supervisor Agent")[1:]

    for turn in turns:
        full_turn_text = "Response from Supervisor Agent" + turn
        
        # Extract the Supervisor's response first
        supervisor_match = patterns['Supervisor'].search(full_turn_text)
        if supervisor_match:
            content = supervisor_match.group(1).strip()
            conversation_step = {
                "step": step_counter,
                "agent_name": "Supervisor Agent",
                "agent_role": "Supervisor Agent",
                "content": content,
                "phase": "Supervising"
            }
            conversation_history.append(conversation_step)
            step_counter += 1

        # Extract any code execution that follows the supervisor
        execution_match = patterns['Execution'].search(full_turn_text)
        if execution_match:
            content = execution_match.group(1).strip()
            conversation_step = {
                "step": step_counter,
                "agent_name": "System",
                "agent_role": "Executor",
                "content": content,
                "phase": "Execution"
            }
            conversation_history.append(conversation_step)
            step_counter += 1
            
        # Extract all responses from the spotify agent within this turn
        spotify_matches = patterns['spotify'].finditer(full_turn_text)
        for match in spotify_matches:
            content = match.group(1).strip()
            conversation_step = {
                "step": step_counter,
                "agent_name": "spotify Agent",
                "agent_role": "spotify Agent",
                "content": content,
                "phase": "Tool Interaction"
            }
            conversation_history.append(conversation_step)
            step_counter += 1


    return {"conversation_history": conversation_history}

def parse_hyperagent_traj_to_conversation_history(log_string: str) -> dict:
    """
    Parses a HyperAgent trajectory log string into a structured multi-turn
    conversation history.

    Args:
        log_string: The raw HyperAgent log file content as a string.

    Returns:
        A dictionary containing the "conversation_history" list.
    """
    conversation_history = []
    step_counter = 1

    # Split the log by "Planner's Response:" which indicates a new high-level step
    planner_turns = log_string.split("HyperAgent_pydata__xarray-4493 - INFO - Planner's Response:")
    
    # The first part is initialization, so we skip it
    for turn_block in planner_turns[1:]:
        turn_block = turn_block.strip()
        
        # The content of the planner's turn is everything up to the next agent's turn
        planner_content_match = re.search(r"(.*?)(?=\n  HyperAgent_pydata__xarray-4493 - INFO -)", turn_block, re.DOTALL)
        
        if not planner_content_match:
            continue
            
        planner_content = planner_content_match.group(1).strip()
        
        # Add the Planner's step
        planner_step = {
            "step": step_counter,
            "agent_name": "Planner",
            "agent_role": "Planner",
            "content": planner_content,
            "phase": "Planning"
        }
        conversation_history.append(planner_step)
        step_counter += 1

        # The rest of the block contains the intern's work and final report to the planner
        intern_work_block = turn_block[planner_content_match.end():]
        
        # Extract the intern's name
        intern_name_match = re.search(r"Intern Name: (.*?)\n", planner_content)
        intern_name = intern_name_match.group(1).strip() if intern_name_match else "Intern Agent"

        # Consolidate all inner-agent responses and the final report back to the planner
        intern_actions = re.findall(
            r"INFO - (Inner-.*?|Navigator->Planner|Editor->Planner|Executor->Planner): (.*?)(?=\n  HyperAgent_pydata__xarray-4493 - INFO -|$)",
            intern_work_block,
            re.DOTALL
        )
        
        if intern_actions:
            full_intern_content = []
            for agent_title, content in intern_actions:
                # Clean up the agent title for clarity
                clean_title = agent_title.replace("Inner-", "").replace("-Assistant", "")
                full_intern_content.append(f"--- Response from {clean_title} ---\n{content.strip()}")

            intern_step = {
                "step": step_counter,
                "agent_name": intern_name,
                "agent_role": intern_name,
                "content": "\n\n".join(full_intern_content),
                "phase": "Execution"
            }
            conversation_history.append(intern_step)
            step_counter += 1
            
    return {"conversation_history": conversation_history}

INPUT_PATH = "datasets/MAD_full_dataset.json"
OUTPUT_JSON = "data_processing/unified_dataset/unified_training_dataset_MAD.json"

# ---------- Helpers for trajectory parsing ----------
# Generic role line like "RoleName: content" with optional timestamp prefix
ROLE_LINE_RE = re.compile(r"^(?:\[\d{4}-.*?\]\s*)?([A-Za-z0-9_][A-Za-z0-9_ \-]+?):\s*(.*)")
MAGENTIC_ONE_BLOCK_PATTERN = re.compile(r"^-{10,}\s*(?:TextMessage|MultiModalMessage)\s*\(([^)]+)\)\s*-{10,}\s*$")

# Common non-dialogue prefixes to ignore when parsing free-form logs
INFRA_PREFIXES: tuple = (
    "=== ",
    "ACTION:",
    "CONTENT:",
    "FROM:",
    "TO:",
    "System:",
    "Parameter |",
    "| --- |",
)

KNOWN_CHATDEV_ROLES = {
    "Programmer", "Code Reviewer", "Chief Technology Officer", "Chief Executive Officer",
    "Counselor", "Chief Product Officer", "CEO", "CTO", "Reviewer", "Tester"
}


def safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def is_all_zero_annotation(mast_annotation: Dict[str, Any]) -> bool:
    if not isinstance(mast_annotation, dict):
        return True
    for v in mast_annotation.values():
        try:
            if float(v) != 0:
                return False
        except Exception:
            # any non-numeric treat as non-zero to keep sample
            return False
    return True


def extract_query_from_text(text: str) -> str:
    # Try to capture a Task: "..." line
    m = re.search(r"Task:\s*\"([^\"]+)\"", text)
    if m:
        return m.group(1)
    # Fallback to first non-empty sentence-like line
    for line in text.splitlines():
        s = line.strip()
        if len(s) > 20 and not s.startswith("[") and not s.startswith("|"):
            return s[:500]
    return ""


def extract_chatdev_conversation(raw_text: str) -> List[Dict[str, Any]]:
    conversation: List[Dict[str, Any]] = []
    lines = raw_text.splitlines()
    current_role = None
    current_buf: List[str] = []

    def flush():
        if current_role is not None and current_buf:
            content = "\n".join(current_buf).strip()
            if content:
                conversation.append({
                    "agent_name": current_role,
                    "agent_role": current_role,
                    "content": content,
                    "phase": "discussion"
                })

    for line in lines:
        s = line.strip()
        # Skip infra/meta lines
        if not s or s.startswith(INFRA_PREFIXES):
            if current_role is not None and s:
                current_buf.append(line)
            continue
        # Role header like "Role: ..."
        m = ROLE_LINE_RE.match(s)
        if m:
            role = m.group(1).strip()
            flush()
            current_role = role
            first = m.group(2).strip()
            current_buf = [first] if first else []
            continue
        # accumulate
        if current_role is not None:
            current_buf.append(line)

    flush()

    # annotate steps
    for i, step in enumerate(conversation, 1):
        step["step"] = i
    return conversation


def extract_magentic_one_conversation(raw_text: str) -> List[Dict[str, Any]]:
    conversation: List[Dict[str, Any]] = []
    lines = raw_text.splitlines()
    current_role = None
    current_buf: List[str] = []

    def flush():
        if current_role is not None and current_buf:
            content = "\n".join(current_buf).strip()
            if content:
                conversation.append({
                    "agent_name": current_role,
                    "agent_role": current_role,
                    "content": content,
                    "phase": "discussion"
                })

    for line in lines:
        m = MAGENTIC_ONE_BLOCK_PATTERN.match(line.strip())
        if m:
            flush()
            current_role = m.group(1).strip()
            current_buf = []
            continue
        if current_role is not None:
            current_buf.append(line)

    flush()
    for i, step in enumerate(conversation, 1):
        step["step"] = i
    return conversation


def extract_generic_conversation(raw_text: str) -> List[Dict[str, Any]]:
    # Fallback generic role: content parsing
    conversation: List[Dict[str, Any]] = []
    lines = raw_text.splitlines()
    current_role = None
    current_buf: List[str] = []

    def flush():
        if current_role is not None and current_buf:
            content = "\n".join(current_buf).strip()
            if content:
                conversation.append({
                    "agent_name": current_role,
                    "agent_role": current_role,
                    "content": content,
                    "phase": "discussion"
                })

    for line in lines:
        s = line.strip()
        if not s or s.startswith(INFRA_PREFIXES):
            if current_role is not None and s:
                current_buf.append(line)
            continue
        m = ROLE_LINE_RE.match(s)
        if m:
            flush()
            current_role = m.group(1).strip()
            first = m.group(2).strip()
            current_buf = [first] if first else []
            continue
        if current_role is not None:
            current_buf.append(line)

    flush()
    for i, step in enumerate(conversation, 1):
        step["step"] = i
    return conversation


def extract_query_and_conversation(sample: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], str]:
    # Try structured fields first
    query = sample.get("question") or sample.get("task") or sample.get("task_prompt") or ""
    final_output = "No final output available"

    # Structured histories
    if isinstance(sample.get("messages"), list):
        conversation = []
        for i, msg in enumerate(sample["messages"], 1):
            role = str(msg.get("role", "unknown")).strip()
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            conversation.append({
                "step": i,
                "agent_name": role.title(),
                "agent_role": role.title(),
                "content": content,
                "phase": "discussion"
            })
        # final_output from last assistant/user
        for msg in reversed(sample["messages"]):
            if str(msg.get("content", "")).strip():
                final_output = str(msg.get("content")).strip()
                break
        if not query:
            # best-effort: first user-like content
            for msg in sample["messages"]:
                if str(msg.get("role", "")).lower() in ("user", "system"):
                    query = str(msg.get("content", ""))
                    break
        return query, conversation, final_output

    if isinstance(sample.get("history"), list):
        conversation = []
        for i, h in enumerate(sample["history"], 1):
            role = str(h.get("role", "unknown")).strip()
            content = str(h.get("content", "")).strip()
            if not content:
                continue
            conversation.append({
                "step": i,
                "agent_name": role.title(),
                "agent_role": role.title(),
                "content": content,
                "phase": "discussion"
            })
        if not query:
            for h in sample["history"]:
                if str(h.get("role", "")).lower() in ("user", "system"):
                    query = str(h.get("content", ""))
                    break
        if conversation:
            final_output = conversation[-1]["content"]
        return query, conversation, final_output

    # Unstructured: prefer explicit trace/trajectory; otherwise collect long text fields
    raw_text = ""
    tr = sample.get("trace")
    if isinstance(tr, dict) and isinstance(tr.get("trajectory"), str) and tr["trajectory"].strip():
        raw_text = tr["trajectory"]
    elif isinstance(sample.get("trace"), str) and sample["trace"].strip():
        raw_text = sample["trace"]
    elif isinstance(sample.get("trajectory"), str) and sample["trajectory"].strip():
        raw_text = sample["trajectory"]
    else:
        long_text_fields: List[str] = []
        for k, v in sample.items():
            if isinstance(v, str) and len(v) > 500:
                long_text_fields.append(v)
        raw_text = "\n\n".join(long_text_fields)

    mas_name = str(sample.get("mas_name", "")).lower()

    # Prefer dedicated parsers if available
    if mas_name in ("chatdev",):
        conv = parse_chatdev_log_to_conversation_history(raw_text).get("conversation_history", [])
    elif mas_name in ("metagpt",):
        conv = parse_metagpt_traj_to_conversation_history(raw_text).get("conversation_history", [])
    elif mas_name in ("openmanus", "open-manus", "open_manus"):
        conv = parse_openmanus_traj_to_conversation_history(raw_text).get("conversation_history", [])
    elif mas_name in ("magentic-one", "magnetic-one", "magenticone", "magentic_one", "magentic"):
        conv = parse_magentic_traj_to_conversation_history(raw_text).get("conversation_history", [])
    elif mas_name in ("ag2",):
        conv = parse_ag2_traj_to_conversation_history(raw_text).get("conversation_history", [])
    elif mas_name in ("appworld",):
        conv = parse_appworld_traj_to_conversation_history(raw_text).get("conversation_history", [])
    elif mas_name in ("hyperagent",):
        conv = parse_hyperagent_traj_to_conversation_history(raw_text).get("conversation_history", [])
    else:
        # Fallbacks
        conv = extract_chatdev_conversation(raw_text)

    if not query:
        query = extract_query_from_text(raw_text)
    if conv:
        final_output = conv[-1]["content"]
    return query, conv, final_output


def convert_sample(idx: int, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    mast = sample.get("mast_annotation", {})
    if is_all_zero_annotation(mast):
        return None

    framework = str(sample.get("mas_name", "mad")).strip().lower()
    # Normalize framework naming to match unified schema's expectation where possible
    framework_map = {
        "chatdev": "chatdev",
        "magentic-one": "magentic_one",
        "magnetic-one": "magentic_one",
        "magenticone": "magentic_one",
        "magentic_one": "magentic_one",
        "metagpt": "metagpt",
    }
    framework_std = framework_map.get(framework, framework)

    query, conversation, final_output = extract_query_and_conversation(sample)

    num_agents = len({step.get("agent_name", "") for step in conversation if step.get("agent_name")}) or 2

    metadata = {
        "framework": framework_std,
        "benchmark": sample.get("benchmark", "MAD"),
        "model": sample.get("model", "unknown"),
        "num_agents": num_agents,
        "num_injected_agents": 0,
        "task_type": sample.get("task_type", "reasoning")
    }

    input_obj = {
        "query": query,
        "conversation_history": conversation,
        "final_output": final_output or "No final output available"
    }

    # Build error_type from mast_annotation (values == 1)
    error_types: List[str] = []
    if isinstance(mast, dict):
        for k, v in mast.items():
            try:
                if int(v) == 1:
                    et = str(k)
                    if not et.startswith("FM-"):
                        et = f"FM-{et}"
                    error_types.append(et)
            except Exception:
                continue
    error_type_str = ", ".join(sorted(error_types)) if error_types else ""

    output_obj = {
        "faulty_agents": [
            {
                "agent_name": "None",
                "error_type": error_type_str,
                "injection_strategy": "prompt_injection",
            }
        ]
    }

    gt_obj = {
        "correct_answer": sample.get("answer", ""),
        "injected_agents": [],
        "is_injection_successful": True
    }

    # Create id: {benchmark}_{model}_{framework}_{hash}_{idx}
    import hashlib
    hash_input = (query or "") + framework_std + (sample.get("id", str(idx)))
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    uid = f"{metadata['benchmark']}_{metadata['model']}_{metadata['framework']}_{file_hash}_{idx}"

    return {
        "id": uid,
        "metadata": metadata,
        "input": input_obj,
        "output": output_obj,
        "ground_truth": gt_obj
    }


def load_mad_dataset(path: str) -> List[Dict[str, Any]]:
    # Try JSON array first
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
    except Exception:
        pass

    # Fallback: JSONL
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                # skip non-json lines
                continue
    return records


def main():
    base = Path(OUTPUT_JSON).parent
    base.mkdir(parents=True, exist_ok=True)

    samples = load_mad_dataset(INPUT_PATH)
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for i, s in enumerate(samples):
        conv = convert_sample(i, s)
        if conv is None:
            dropped += 1
            continue
        kept.append(conv)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"Total samples: {len(samples)}")
    print(f"Kept (with mistakes): {len(kept)}")
    print(f"Dropped (all-zero annotations): {dropped}")
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()