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
import random
import json
import argparse
import os


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
        subset: Optional[int] = None,
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
        if subset:
            datas = datas[:subset]
        
        if idx is not None:
            # pick only the tasks with the specified idx
            if len(idx) != 0:   
                datas = [datas[i] for i in idx]
                
        return datas

async def run_task_and_capture_log(team, task_prompt):
    # 捕获 Console 的输出
    from io import StringIO
    import sys

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        print(f"Running task: {task_prompt}")
        await Console(team.run_stream(task=task_prompt))
    finally:
        sys.stdout = old_stdout
    return mystdout.getvalue()

async def main() -> None:
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="GAIA 测试脚本")
    parser.add_argument("--level", type=int, default=2, help="GAIA level (1, 2, 3)")
    parser.add_argument("--on", type=str, default="valid", choices=["valid", "test"], help="数据集类型")
    parser.add_argument("--use-subset", action="store_true", help="使用子集数据集")
    parser.add_argument("--subset-file", type=str, default="owl/data/gaia_subset.json", help="子集文件路径")
    parser.add_argument("--subset-size", type=int, help="限制子集大小")
    parser.add_argument("--randomize", action="store_true", help="随机化任务顺序")
    
    args = parser.parse_args()

    # 创建独立的 model clients 避免共享
    def create_model_client():
        return OpenAIChatCompletionClient(
            model="gpt-4o-mini-2024-07-18",
            api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
    
    LEVEL = args.level
    on = args.on
    if args.use_subset:
        OUTPUT_PATH = f"magentic_one/logs/level_{LEVEL}_{on}_subset.json"
    else:
        OUTPUT_PATH = f"magentic_one/logs/level_{LEVEL}_{on}.json"
    
    # 加载任务
    tasks = load_tasks(
        on=on, 
        level=LEVEL,
        randomize=args.randomize,
        subset=args.subset_size,
        use_subset=args.use_subset,
        subset_file=args.subset_file
    )
    
    print(f"📋 加载了 {len(tasks)} 个任务")
    if len(tasks) > 0:
        print(f"🔍 第一个任务ID: {tasks[0]['task_id']}")

    # === 断点重续逻辑保持不变 ===
    import os
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            all_logs = json.load(f)
    else:
        all_logs = {}

    # === 修改：每个任务创建新的团队，每个 agent 使用独立的 model client ===
    for task in tasks:
        task_id = task["task_id"]
        if task_id in all_logs:
            print(f"Task {task_id} 已完成，跳过。")
            continue
            
        # 为每个 agent 创建独立的 model client
        surfer_client = create_model_client()
        file_surfer_client = create_model_client()
        coder_client = create_model_client()
        team_client = create_model_client()
        
        # 为每个任务创建新的团队实例，使用独立的 clients
        surfer = MultimodalWebSurfer("WebSurfer", model_client=surfer_client)
        file_surfer = FileSurfer("FileSurfer", model_client=file_surfer_client)
        coder = MagenticOneCoderAgent("Coder", model_client=coder_client)
        terminal = CodeExecutorAgent("ComputerTerminal", code_executor=LocalCommandLineCodeExecutor())
        team = MagenticOneGroupChat([surfer, file_surfer, coder, terminal], model_client=team_client)

        question = task["Question"]
        answer = task["Final answer"]
        all_logs[task_id] = {}
        all_logs[task_id]["question"] = question
        all_logs[task_id]["correct_answer"] = answer
        all_logs[task_id]["logs"] = await run_task_and_capture_log(team, question)

        # 每做完一个就保存一次
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_logs, f, ensure_ascii=False, indent=2)
        
        print(f"Task {task_id} 完成")

    print(f"全部任务处理完成，日志已保存到 {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())