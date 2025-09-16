import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from typing import Any, Dict, List, Literal, Optional, Union, Tuple, Callable
from injection import MagenticInjectionCoordinator
import random
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--inject", default=True, help="Enable malicious injection.")
parser.add_argument("--target-agent", type=str, default="WebSurfer", choices=["Orchestrator", "WebSurfer", "FileSurfer", "Coder", "ComputerTerminal"], help="The agent to inject.")
parser.add_argument("--fm-type", type=str, default="FM-2.5", help="The FM error type to inject.")
parser.add_argument("--injection-strategy", type=str, choices=["prompt_injection", "response_corruption"], default="prompt_injection", help="The injection strategy.")
# 添加子集相关参数
parser.add_argument("--randomize", action="store_true", help="随机化任务顺序")
parser.add_argument("--level", type=str, default="all", help="GAIA level: 1, 2, 3, 或 'all' 表示所有级别")
parser.add_argument("--on", type=str, default="valid", choices=["valid", "test"], help="数据集类型")
# 日志捕获相关参数
parser.add_argument("--capture-mode", type=str, default="stream", 
                    choices=["stream", "memory", "direct", "none"],
                    help="日志捕获模式: stream(流式缓存,推荐), memory(内存优化), direct(直接文件), none(不捕获)")
parser.add_argument("--flush-threshold", type=int, default=8192, help="流式模式的缓存阈值(字节)")
parser.add_argument("--limit", type=int, default=None, help="限制处理的任务数量，用于批量数据收集")


def create_model_client():
        return OpenAIChatCompletionClient(
            model="gpt-4o-mini-2024-07-18",
            api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE"),  # Use environment variable
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")  # Use environment variable
        )

def load(force_download=False):
        r"""Load the GAIA dataset.

        Args:
            force_download (bool, optional): Whether to
                force download the data.
        """
        # Define validation and test directories - use relative paths or environment variables
        valid_dir = os.getenv("GAIA_VALIDATION_DIR", "./data/gaia/2023/validation")
        test_dir = os.getenv("GAIA_TEST_DIR", "./data/gaia/2023/test")

        all_data = {}
        # Load metadata for both validation and test datasets
        for path, label in zip([valid_dir, test_dir], ["valid", "test"]):
            all_data[label] = []
            with open(f"{path}/metadata.jsonl", "r") as f:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    if data["task_id"] == "0-0-0-0-0":
                        continue
                    if data["file_name"]:
                        data["file_name"] = f"{path}/{data['file_name']}"
                    all_data[label].append(data)
        return all_data

def load_tasks(
        on: Literal["valid", "test"],
        level: Union[int, List[int], Literal["all"]],
        randomize: bool = False,
        idx: Optional[List[int]] = None,
        use_subset: bool = False,
        subset_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        r"""Load tasks from the dataset."""
        gaia_data = load()
        if on not in ["valid", "test"]:
            raise ValueError(
                f"Invalid value for `on`: {on}, expected 'valid' or 'test'."
            )
        levels = (
            [1, 2, 3]
            if level == "all"
            else [level]
            if isinstance(level, int)
            else level
        )   
        
        datas = [data for data in gaia_data[on] if data["Level"] in levels]
        
        # 如果使用子集，过滤任务
        if use_subset and subset_file:
            if os.path.exists(subset_file):
                with open(subset_file, 'r', encoding='utf-8') as f:
                    subset_data = json.load(f)
                    subset_task_ids = set(subset_data.get("task_ids", []))
                    print(f"📋 使用子集数据集: {subset_file}")
                    print(f"📊 子集包含 {len(subset_task_ids)} 个任务")
                    
                    # 过滤数据
                    original_count = len(datas)
                    datas = [data for data in datas if data["task_id"] in subset_task_ids]
                    print(f"🔍 过滤后剩余 {len(datas)} 个任务 (从 {original_count} 个)")
            else:
                print(f"⚠️  子集文件不存在: {subset_file}")
        
        if randomize:
            random.shuffle(datas)
        
        if idx is not None:
            # pick only the tasks with the specified idx
            if len(idx) != 0:   
                datas = [datas[i] for i in idx]
                
        return datas

async def run_task_and_capture_log(team, task_prompt, capture_mode="stream", max_log_size=None):
    """
    优化的日志捕获函数 - 保证完整输出的同时优化性能
    
    Args:
        capture_mode: 
            - "stream": 流式写入文件，保持完整日志(推荐)
            - "memory": 内存缓存，带性能优化
            - "direct": 直接文件写入，最节省内存
            - "none": 不捕获日志
    """
    import sys
    import gc
    import tempfile
    import os
    import threading
    from collections import deque
    
    if capture_mode == "none":
        print(f"Running task: {task_prompt}")
        await Console(team.run_stream(task=task_prompt))
        return f"Task completed: {task_prompt[:100]}..."
    
    elif capture_mode == "direct":
        # 直接写入文件，内存占用最小
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log', encoding='utf-8') as temp_file:
            old_stdout = sys.stdout
            sys.stdout = temp_file
            
            try:
                print(f"Running task: {task_prompt}")
                await Console(team.run_stream(task=task_prompt))
            finally:
                sys.stdout = old_stdout
                temp_file.flush()
                
                # 读取完整文件内容
                with open(temp_file.name, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                
                # 删除临时文件
                os.unlink(temp_file.name)
                gc.collect()
        
        return log_content
    
    elif capture_mode == "stream":
        # 流式缓存，定期刷新到磁盘，平衡性能和完整性
        class StreamingFileBuffer:
            def __init__(self, flush_threshold=8192):  # 8KB 缓存
                self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log', encoding='utf-8')
                self.buffer = []
                self.buffer_size = 0
                self.flush_threshold = flush_threshold
                self.lock = threading.Lock()
            
            def write(self, s):
                with self.lock:
                    self.buffer.append(s)
                    self.buffer_size += len(s)
                    
                    # 达到阈值时刷新到文件
                    if self.buffer_size >= self.flush_threshold:
                        self._flush_buffer()
                
                return len(s)
            
            def _flush_buffer(self):
                if self.buffer:
                    self.temp_file.write(''.join(self.buffer))
                    self.temp_file.flush()
                    self.buffer.clear()
                    self.buffer_size = 0
            
            def flush(self):
                with self.lock:
                    self._flush_buffer()
                    self.temp_file.flush()
            
            def getvalue(self):
                with self.lock:
                    self._flush_buffer()
                    self.temp_file.seek(0)
                    content = self.temp_file.read()
                    self.temp_file.close()
                    
                    # 删除临时文件
                    try:
                        os.unlink(self.temp_file.name)
                    except:
                        pass
                    
                    return content

        old_stdout = sys.stdout
        stream_buffer = StreamingFileBuffer()
        sys.stdout = stream_buffer
        
        try:
            print(f"Running task: {task_prompt}")
            await Console(team.run_stream(task=task_prompt))
        finally:
            sys.stdout = old_stdout
            log_content = stream_buffer.getvalue()
            gc.collect()
        
        return log_content
    
    else:  # capture_mode == "memory"
        # 优化的内存捕获，使用 deque 提高性能
        class OptimizedStringIO:
            def __init__(self):
                self.chunks = deque()
                self.total_size = 0
            
            def write(self, s):
                if s:
                    self.chunks.append(s)
                    self.total_size += len(s)
                return len(s)
            
            def getvalue(self):
                if not self.chunks:
                    return ""
                result = ''.join(self.chunks)
                # 清理内存
                self.chunks.clear()
                self.total_size = 0
                return result
            
            def flush(self):
                pass

        old_stdout = sys.stdout
        mystdout = OptimizedStringIO()
        sys.stdout = mystdout
        
        try:
            print(f"Running task: {task_prompt}")
            await Console(team.run_stream(task=task_prompt))
        finally:
            sys.stdout = old_stdout
            log_content = mystdout.getvalue()
            del mystdout  # 显式删除
            gc.collect()
        
        return log_content

async def main() -> None:
    args = parser.parse_args()

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini-2024-07-18",
        api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    injection_coordinator = MagenticInjectionCoordinator(args, model_client)
    # print(f"🔍 DEBUG: injection_coordinator created")
    # print(f"🔍 DEBUG: injection_coordinator.enabled = {injection_coordinator.enabled}")
    # print(f"🔍 DEBUG: injection_coordinator.target_agent_name = {injection_coordinator.target_agent_name}")
    # print(f"🔍 DEBUG: injection_coordinator.fm_error_type = {injection_coordinator.fm_error_type}")
    # print(f"🔍 DEBUG: injection_coordinator.injection_strategy = {injection_coordinator.injection_strategy}")
    
    # 处理级别参数
    if args.level == "all":
        LEVEL = "all"
        level_for_loading = "all"
        subset_file_path = os.getenv("GAIA_SUBSET_DIR", "./data/gaia_subset") + "/gaia_subset_all_levels.json"
    else:
        try:
            LEVEL = int(args.level)
            level_for_loading = LEVEL
            subset_file_path = os.getenv("GAIA_SUBSET_DIR", "./data/gaia_subset") + f"/gaia_subset_level_{args.level}.json"
        except ValueError:
            raise ValueError(f"无效的级别参数: {args.level}. 请使用 1, 2, 3 或 'all'")
    
    on = args.on
    OUTPUT_PATH = f"magentic_one/injection_logs/level_{LEVEL}_{on}_{args.target_agent}_{args.fm_type}_{args.injection_strategy}.json"

    # 加载任务
    tasks = load_tasks(
        on=on, 
        level=level_for_loading,
        randomize=args.randomize,
        use_subset=True,
        subset_file=subset_file_path
    )
    
    print(f"📋 加载了 {len(tasks)} 个任务")
    if len(tasks) > 0:
        print(f"🔍 第一个任务ID: {tasks[0]['task_id']}")

    # === 断点重续逻辑保持不变 ===
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            all_logs = json.load(f)
    else:
        all_logs = {}

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini-2024-07-18",
        api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    # === 修改：每个任务创建新的团队 ===
    completed_count = 0
    for task in tasks:
        task_id = task["task_id"]
        if task_id in all_logs:
            print(f"Task {task_id} 已完成，跳过。")
            continue
        
        # 检查是否达到限制
        if args.limit is not None and completed_count >= args.limit:
            print(f"🎯 已完成 {completed_count} 个任务，达到限制 ({args.limit})，停止处理。")
            break
        
        surfer_client = create_model_client()
        file_surfer_client = create_model_client()
        coder_client = create_model_client()
        team_client = create_model_client()
        
        # 为每个任务创建新的团队实例
        surfer = MultimodalWebSurfer("WebSurfer", model_client=surfer_client)
        file_surfer = FileSurfer("FileSurfer", model_client=file_surfer_client)
        coder = MagenticOneCoderAgent("Coder", model_client=coder_client)
        terminal = CodeExecutorAgent("ComputerTerminal", code_executor=LocalCommandLineCodeExecutor())
        team = MagenticOneGroupChat([surfer, file_surfer, coder, terminal], model_client=team_client)
        
        # print(f"🔍 DEBUG: Created team for task {task_id}")
        # print(f"🔍 DEBUG: team.model_client id = {id(team._model_client) if hasattr(team, '_model_client') else 'NO_MODEL_CLIENT'}")
        # print(f"🔍 DEBUG: injection model_client id = {id(model_client)}")
        
        # 应用注入（如果目标不是Orchestrator，立即注入；否则延迟注入）
        injection_coordinator.apply_injection(team)
        
        # 如果目标agent是Orchestrator，需要在运行时注入
        if injection_coordinator.enabled and injection_coordinator.target_agent_name == "Orchestrator":
            # 创建一个包装函数来在运行时注入
            original_run_stream = team.run_stream
            
            async def run_stream_with_injection(*args, **kwargs):
                # print("🔍 DEBUG: run_stream_with_injection called!")
                # print(f"🔍 DEBUG: hasattr(team, '_group_chat_manager') = {hasattr(team, '_group_chat_manager')}")
                # print(f"🔍 DEBUG: team._group_chat_manager = {getattr(team, '_group_chat_manager', 'NOT_FOUND')}")
                
                # 确保在每次运行时注入都是活跃的
                # print("🎯 Runtime injection: Applying injection to ensure all LLM calls are intercepted...")
                injection_coordinator.apply_injection(team)
                # 返回异步生成器，而不是协程
                async for message in original_run_stream(*args, **kwargs):
                    yield message
            team.run_stream = run_stream_with_injection

        question = task["Question"]
        answer = task["Final answer"]
        all_logs[task_id] = {}
        all_logs[task_id]["question"] = question
        all_logs[task_id]["correct_answer"] = answer
        all_logs[task_id]["logs"] = await run_task_and_capture_log(
            team, question, 
            capture_mode=args.capture_mode
        )

        # 每做完一个就保存一次
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_logs, f, ensure_ascii=False, indent=2)
        
        completed_count += 1
        print(f"Task {task_id} 完成 ({completed_count}/{args.limit if args.limit else len(tasks)})")

    print(f"全部任务处理完成，日志已保存到 {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())