# EXAMPLE COMMAND: from folder examples/open_deep_research, run: python run_gaia.py --concurrency 32 --run-name generate-traces-03-apr-noplanning --model-id gpt-4o
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from scripts.reformulator import prepare_response
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
    SearchInformationTool,
)
from scripts.resilient_web_search_tool import ResilientWebSearchTool
from scripts.visual_qa import visualizer
from tqdm import tqdm


class GAIAJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for GAIA results that handles ChatMessage objects"""
    
    def default(self, obj):
        # Handle ChatMessage objects
        if hasattr(obj, '__class__') and 'ChatMessage' in obj.__class__.__name__:
            return {
                'type': 'ChatMessage',
                'role': getattr(obj, 'role', None),
                'content': str(getattr(obj, 'content', ''))
            }
        
        # Handle other objects that might not be serializable
        if hasattr(obj, '__dict__'):
            return str(obj)
        
        # Let the base class default method raise the TypeError
        return super().default(obj)

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    Model,
    ToolCallingAgent,
    InferenceClientModel,
)


# Optional: FM malicious injection (only used if enabled via CLI)
try:
    from gaia_agents.magentic_one.malicious_factory.fm_malicious_system import (
        FMMaliciousFactory,
        FMErrorType,
        InjectionStrategy,
        AgentContext,
    )
except Exception:
    try:
        project_root = Path(__file__).resolve().parents[5]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from gaia_agents.magentic_one.malicious_factory.fm_malicious_system import (
            FMMaliciousFactory,
            FMErrorType,
            InjectionStrategy,
            AgentContext,
        )
    except Exception:
        FMMaliciousFactory = None  # type: ignore
        FMErrorType = None  # type: ignore
        InjectionStrategy = None  # type: ignore
        AgentContext = None  # type: ignore

load_dotenv(override=True)
# Avoid interactive HF login; only login if a valid-looking token is provided
_hf_token = os.getenv("HF_TOKEN", "")
if _hf_token and _hf_token.startswith("hf_"):
    try:
        login(_hf_token)
    except Exception:
        print("HF login failed with provided token; continuing without HF login.")
else:
    print("HF_TOKEN not set to a valid token; skipping Hugging Face login.")

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="o1")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--set-to-run", type=str, default="validation")
    parser.add_argument("--use-open-models", type=bool, default=False)
    parser.add_argument("--open-provider", type=str, default="novita")
    parser.add_argument("--open-model-id", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--use-raw-dataset", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    # Injection switches
    parser.add_argument("--enable-injection", action="store_true")
    parser.add_argument("--fm-type", type=str, default="FM-1.1")
    parser.add_argument(
        "--injection-strategy",
        type=str,
        default="prompt_injection",
        choices=["prompt_injection", "response_corruption"],
    )
    return parser.parse_args()


### IMPORTANT: EVALUATION SWITCHES

print("Make sure you deactivated any VPN like Tailscale, else some URLs will be blocked!")

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}


user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 2048,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 120,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent_team(model: Model):
    text_limit = 40000
    ti_tool = TextInspectorTool(model, text_limit)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    # 统一使用弹性搜索工具，内部优先 Serper，失败回退 SerpAPI；同时提供文本浏览器搜索
    resilient_web_search = ResilientWebSearchTool()
    search_info_tool = SearchInformationTool(browser)
    setattr(search_info_tool, "name", "text_browser_search")

    WEB_TOOLS = [
        resilient_web_search,
        search_info_tool,
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    # Ensure tool names are unique to avoid ValueError in ToolCallingAgent
    unique_tools_by_name = {}
    for tool in WEB_TOOLS:
        tool_name = getattr(tool, "name", None)
        if tool_name and tool is not None and tool_name not in unique_tools_by_name:
            unique_tools_by_name[tool_name] = tool
    WEB_TOOLS = list(unique_tools_by_name.values())

    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=12,
        verbosity_level=1,
        planning_interval=6,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=False,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool],
        max_steps=8,
        verbosity_level=1,
        additional_authorized_imports=["*"],
        planning_interval=6,
        managed_agents=[text_webbrowser_agent],
    )
    return manager_agent


def load_gaia_dataset(use_raw_dataset: bool, set_to_run: str) -> datasets.Dataset:
    if not os.path.exists("data/gaia"):
        if use_raw_dataset:
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )
        else:
            # WARNING: this dataset is gated: make sure you visit the repo to require access.
            snapshot_download(
                repo_id="smolagents/GAIA-annotated",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )

    def preprocess_file_paths(row):
        if len(row["file_name"]) > 0:
            row["file_name"] = f"data/gaia/{set_to_run}/" + row["file_name"]
        return row

    eval_ds = datasets.load_dataset(
        "gaia-benchmark/GAIA",
        name="2023_all",
        split=set_to_run,
        # data_files={"validation": "validation/metadata.jsonl", "test": "test/metadata.jsonl"},
    )

    eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})
    eval_ds = eval_ds.map(preprocess_file_paths)
    return eval_ds


def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_path = Path(jsonl_file)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, cls=GAIAJSONEncoder) + "\n")
    assert jsonl_path.exists(), "File not found!"
    print("Answer exported to file:", jsonl_path.resolve())


def _sanitize_steps(steps: list[dict]) -> list[dict]:
    """Sanitize step dicts to avoid huge/binary payloads in JSONL.

    - Drops observations_images (can be large binary bytes)
    - Keeps ChatMessage objects; GAIAJSONEncoder will stringify their content
    """
    sanitized: list[dict] = []
    for step in steps or []:
        try:
            s = dict(step)
            if "observations_images" in s and s["observations_images"] is not None:
                s["observations_images"] = None
            sanitized.append(s)
        except Exception:
            # Fallback: keep raw step
            sanitized.append(step)
    return sanitized


def answer_single_question(
    example: dict,
    model_id: str,
    answers_file: str,
    visual_inspection_tool: TextInspectorTool,
    use_open_models: bool = False,
    open_provider: str | None = None,
    open_model_id: str | None = None,
    injection_enabled: bool = False,
    fm_type: str | None = None,
    injection_strategy: str | None = None,
    injection_target_agent: str | None = "manager",
    injection_target_agent_index: int | None = None,
) -> None:
    model_params: dict[str, Any] = {
        "model_id": model_id,
        "custom_role_conversions": custom_role_conversions,
    }
    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
        model_params["max_completion_tokens"] = 8192
    else:
        model_params["max_tokens"] = 4096
    if use_open_models:
        omid = open_model_id or "Qwen/Qwen3-32B"
        oprov = open_provider or "novita"
        model = InferenceClientModel(model_id=omid, provider=oprov, max_tokens=model_params.get("max_tokens", 4096), custom_role_conversions=custom_role_conversions)
    else:
        model = LiteLLMModel(**model_params)
    document_inspection_tool = TextInspectorTool(model, 40000)

    agent = create_agent_team(model)

    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer! Here is the task:

""" + example["question"]

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
            prompt_use_files += get_single_file_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
            )
        augmented_question += prompt_use_files

    # Preserve original augmented question for explicit traceability
    augmented_question_original = augmented_question

    # Inspect search_agent system prompt (original) before any potential injection
    search_agent_system_prompt_original = None
    try:
        _web_agent = list(agent.managed_agents.values())[0]
        if hasattr(_web_agent, "memory") and hasattr(_web_agent.memory, "system_prompt"):
            search_agent_system_prompt_original = _web_agent.memory.system_prompt.system_prompt
    except Exception:
        search_agent_system_prompt_original = None

    # Optional: prompt injection before run
    used_injection = False
    injected_question = None
    fm_used = None
    strategy_used = None
    injection_instruction_text = None
    clean_output_before_corruption = None
    # Explicit injection bookkeeping
    injection_method_id = None
    prompt_injection_applied = False
    response_corruption_applied = False
    injection_insertion_point = None
    if injection_enabled and FMMaliciousFactory is not None and injection_strategy == "prompt_injection":
        try:
            factory = FMMaliciousFactory()
            target_role = (injection_target_agent or "manager").strip()
            ctx = AgentContext(role_name=target_role, description="gaia full run")
            try:
                fm_enum = FMErrorType(fm_type) if fm_type is not None else FMErrorType.FM_1_1
            except Exception:
                fm_enum = FMErrorType.FM_1_1
            # Record the concrete instruction text used for this prompt injection
            try:
                injection_instruction_text = factory.get_injection_instruction(fm_enum, ctx, InjectionStrategy.PROMPT_INJECTION)
            except Exception:
                injection_instruction_text = None
            if (injection_target_agent or "manager") == "manager":
                injected_question = factory.inject_prompt(augmented_question, fm_enum, ctx)
                if injected_question:
                    augmented_question = injected_question
                    used_injection = True
                    fm_used = fm_enum.value
                    strategy_used = "prompt_injection"
                    injection_method_id = f"{fm_enum.value}_prompt_injection"
                    prompt_injection_applied = True
                    injection_insertion_point = "manager.input"
            else:
                # Inject into search_agent system prompt to persist through its whole run
                try:
                    web_agent = list(agent.managed_agents.values())[0]
                except Exception:
                    web_agent = None
                if web_agent is not None and hasattr(web_agent, "memory") and hasattr(web_agent.memory, "system_prompt"):
                    try:
                        original_sp = web_agent.memory.system_prompt.system_prompt
                        modified_sp = factory.inject_prompt(original_sp or "", fm_enum, ctx)
                        if modified_sp and isinstance(modified_sp, str):
                            web_agent.memory.system_prompt.system_prompt = modified_sp
                            used_injection = True
                            fm_used = fm_enum.value
                            strategy_used = "prompt_injection"
                            injection_method_id = f"{fm_enum.value}_prompt_injection"
                            prompt_injection_applied = True
                            injection_insertion_point = "search_agent.system_prompt"
                    except Exception:
                        pass
        except Exception:
            pass

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Run agent 🚀
        final_result = agent.run(augmented_question)

        agent_memory_messages = agent.write_memory_to_messages()

        final_result = prepare_response(augmented_question, agent_memory_messages, reformulation_model=model)

        output = str(final_result)
        # Optional: response corruption after run
        if injection_enabled and FMMaliciousFactory is not None and injection_strategy == "response_corruption":
            try:
                factory = locals().get("factory") or FMMaliciousFactory()
                target_role = (injection_target_agent or "manager").strip()
                ctx = AgentContext(role_name=target_role, description="gaia full run")
                try:
                    fm_enum = FMErrorType(fm_type) if fm_type is not None else FMErrorType.FM_2_4
                except Exception:
                    fm_enum = FMErrorType.FM_2_4
                # Record instruction and clean output before corruption
                try:
                    injection_instruction_text = factory.get_injection_instruction(fm_enum, ctx, InjectionStrategy.RESPONSE_CORRUPTION)
                except Exception:
                    injection_instruction_text = None
                clean_output_before_corruption = output
                corrupted = factory.corrupt_response(output, fm_enum, ctx)
                if corrupted:
                    output = str(corrupted)
                    used_injection = True
                    fm_used = fm_enum.value
                    strategy_used = "response_corruption"
                    injection_method_id = f"{fm_enum.value}_response_corruption"
                    response_corruption_applied = True
                    injection_insertion_point = "manager.output"
            except Exception:
                pass
        # Messages replay (legacy, human-readable)
        intermediate_steps = agent_memory_messages

        # Check for parsing errors which indicate the LLM failed to follow the required format
        parsing_error = any(["AgentParsingError" in str(step) for step in agent.memory.steps])

        # check if iteration limit exceeded
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        print("Error on ", augmented_question, e)
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    token_counts_manager = agent.monitor.get_total_token_counts()
    token_counts_web = list(agent.managed_agents.values())[0].monitor.get_total_token_counts()
    total_token_counts = {
        "input": token_counts_manager.input_tokens + token_counts_web.input_tokens,
        "output": token_counts_manager.output_tokens + token_counts_web.output_tokens,
    }
    # Collect per-agent full steps for detailed logging
    manager_steps_full = []
    manager_system_prompt = None
    search_agent_steps_full = []
    search_agent_system_prompt = None
    try:
        manager_system_prompt = agent.memory.system_prompt.system_prompt
        manager_steps_full = _sanitize_steps(agent.memory.get_full_steps())
    except Exception:
        manager_steps_full = []
    try:
        web_agent = list(agent.managed_agents.values())[0]
        if hasattr(web_agent, "memory"):
            search_agent_system_prompt = web_agent.memory.system_prompt.system_prompt
            search_agent_steps_full = _sanitize_steps(web_agent.memory.get_full_steps())
    except Exception:
        search_agent_steps_full = []
    # Injection target bookkeeping (explicit)
    injection_target_agent_label = (injection_target_agent if injection_enabled else None)
    injection_target_manager = True if injection_target_agent_label == "manager" else False
    injection_target_search = True if injection_target_agent_label == "search_agent" else False

    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "augmented_question_original": augmented_question_original,
        "prediction": output,
        "prediction_after_corruption": output,
        "intermediate_steps": intermediate_steps,
        # Detailed per-agent traces
        "agents": {
            "manager": {
                "agent_type": agent.__class__.__name__,
                "system_prompt": manager_system_prompt,
                "token_counts": {
                    "input": token_counts_manager.input_tokens,
                    "output": token_counts_manager.output_tokens,
                },
                "steps_full": manager_steps_full,
                "injection_target": injection_target_manager,
            },
            "search_agent": {
                "agent_type": list(agent.managed_agents.values())[0].__class__.__name__ if agent.managed_agents else None,
                "system_prompt": search_agent_system_prompt,
                "system_prompt_original": search_agent_system_prompt_original,
                "token_counts": {
                    "input": token_counts_web.input_tokens,
                    "output": token_counts_web.output_tokens,
                },
                "steps_full": search_agent_steps_full,
                "injection_target": injection_target_search,
            },
        },
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "task": example["task"],
        "task_id": example["task_id"],
        "true_answer": example["true_answer"],
        "start_time": start_time,
        "end_time": end_time,
        "token_counts": total_token_counts,
        "injection_enabled": injection_enabled,
        "injection_fm_type": fm_used,
        "injection_strategy": strategy_used,
        "injected_question": injected_question,
        # Injection trace helpers
        "injection_instruction": injection_instruction_text,
        "clean_prediction_before_corruption": clean_output_before_corruption,
        # Explicit injection target for easier downstream analysis
        "injection_target_agent": injection_target_agent_label,
        "injection_target_agent_index": None,
        # New explicit traceability fields
        "injection_method_id": injection_method_id,
        "prompt_injection_applied": prompt_injection_applied,
        "response_corruption_applied": response_corruption_applied,
        "injection_insertion_point": injection_insertion_point,
    }
    append_answer(annotated_example, answers_file)


def get_examples_to_answer(answers_file: str, eval_ds: datasets.Dataset) -> list[dict]:
    print(f"Loading answers from {answers_file}...")
    try:
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        print(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        print("Error when loading records: ", e)
        print("No usable records! ▶️ Starting new.")
        done_questions = []
    return [line for line in eval_ds.to_list() if line["question"] not in done_questions ]


def main():
    args = parse_args()
    print(f"Starting run with arguments: {args}")

    eval_ds = load_gaia_dataset(args.use_raw_dataset, args.set_to_run)
    print("Loaded evaluation dataset:")
    print(pd.DataFrame(eval_ds)["task"].value_counts())

    answers_file = f"output/{args.set_to_run}/{args.run_name}.jsonl"
    tasks_to_run = get_examples_to_answer(answers_file, eval_ds)
    if args.max_tasks is not None and args.max_tasks > 0:
        tasks_to_run = tasks_to_run[: args.max_tasks]

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(
                answer_single_question,
                example,
                args.model_id,
                answers_file,
                visualizer,
                args.use_open_models,
                args.open_provider,
                args.open_model_id,
            )
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()

    # for example in tasks_to_run:
    #     answer_single_question(example, args.model_id, answers_file, visualizer)
    print("All tasks processed.")


if __name__ == "__main__":
    main()
