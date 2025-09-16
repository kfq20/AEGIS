#!/usr/bin/env python3
"""
Gemini Anomaly Detection Script with Multiprocessing
使用 Gemini 模型对数据集进行异常检测评测（支持多进程并行处理）

Usage:
    python model_mea.py --input data_processing/unified_dataset/final_data/whowhen.jsonl --output results_gemini.jsonl
    python model_mea.py --input data_processing/unified_dataset/final_data/whowhen.jsonl --output results_gemini.jsonl --limit 10 --batch_size 4

batch_size 参数控制并行进程数：
- batch_size=4 表示同时启动4个进程，每个进程处理一个不同的数据样本
- 每个进程独立工作，处理完一个样本后会自动获取下一个样本
"""

import json
import argparse
import time
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Process, Queue, Manager
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_mea.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GeminiAnomalyDetector:
    """使用 Gemini 模型进行异常检测的类（支持多进程处理）"""
    
    def __init__(self, model: str, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"):
        """
        初始化 Gemini 客户端
        
        Args:
            api_key: Gemini API 密钥
            base_url: API 基础 URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
    def load_prompt_template(self, prompt_file: str) -> str:
        """
        加载 prompt 模板
        
        Args:
            prompt_file: prompt 文件路径
            
        Returns:
            prompt 模板字符串
        """
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt 文件未找到: {prompt_file}")
            raise
    
    def extract_conversation_text(self, input_data: Dict) -> str:
        """
        从输入数据中提取完整的对话文本，包括 query 和 conversation_history
        
        Args:
            input_data: 输入数据字典，包含 query 和 conversation_history
            
        Returns:
            格式化的完整对话文本
        """
        query = input_data.get('query', '')
        conversation_history = input_data.get('conversation_history', [])
        
        # 开始构建对话文本
        conversation_text = f"QUERY:\n{query}\n\n"
        conversation_text += "CONVERSATION HISTORY:\n"
        
        for entry in conversation_history:
            step = entry.get('step', '')
            agent_name = entry.get('agent_name', '')
            agent_role = entry.get('agent_role', '')
            content = entry.get('content', '')
            phase = entry.get('phase', '')
            
            conversation_text += f"Step {step} - {agent_name} ({agent_role}) [{phase}]:\n{content}\n\n"
        
        return conversation_text.strip()
    
    def detect_anomalies_sync(self, conversation_text: str, max_retries: int = 3) -> Optional[Dict]:
        """
        同步方式使用 Gemini 模型检测异常
        
        Args:
            conversation_text: 对话文本
            max_retries: 最大重试次数
            
        Returns:
            检测结果字典
        """
        prompt_template = self.load_prompt_template("baseline/prompt_cot.txt")
        prompt = prompt_template.format(conversation_text=conversation_text)
        
        # 为每个进程创建独立的客户端
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise JSON response generator. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    # temperature=0
                )
                
                # 检查响应结构是否完整
                if not response.choices or len(response.choices) == 0:
                    logger.warning(f"API 返回空的 choices (尝试 {attempt + 1}/{max_retries})")
                    logger.debug(f"完整响应: {response}")
                    if attempt == max_retries - 1:
                        return {"faulty_agents": []}, ""
                    time.sleep(1)
                    continue
                
                if not response.choices[0].message or not response.choices[0].message.content:
                    logger.warning(f"API 返回空内容 (尝试 {attempt + 1}/{max_retries})")
                    logger.debug(f"响应结构: choices[0]={response.choices[0] if response.choices else None}")
                    if attempt == max_retries - 1:
                        return {"faulty_agents": []}, ""
                    time.sleep(1)
                    continue
                
                response_text = response.choices[0].message.content.strip()
                
                # 尝试解析 JSON
                try:
                    # 移除可能的 markdown 格式
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    
                    result = json.loads(response_text)
                    return result, response_text
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    
                    if attempt == max_retries - 1:
                        # 最后一次尝试，返回默认结果
                        return {"faulty_agents": []}, ""
                    
                    time.sleep(1)  # 等待一秒后重试
                    
            except Exception as e:
                logger.error(f"API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return {"faulty_agents": []}, ""
                time.sleep(2)  # 等待两秒后重试
        
        return {"faulty_agents": []}, ""
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """
        评估单个样本
        
        Args:
            sample: 数据样本
            
        Returns:
            评估结果
        """
        try:
            # 提取完整的对话文本（包括 query 和 conversation_history）
            input_data = sample.get('input', {})
            conversation_text = self.extract_conversation_text(input_data)
            
            # 使用 Gemini 检测异常
            detection_result, response_text = self.detect_anomalies_sync(conversation_text)
            
            # 构建结果
            result = {
                "id": sample.get("id"),
                "metadata": sample.get("metadata"),
                "input": sample.get("input"),
                "ground_truth": sample.get("output"),
                "model_detection": detection_result,
                "response_text": response_text,
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"评估样本失败 {sample.get('id', 'unknown')}: {e}")
            return {
                "id": sample.get("id"),
                "error": str(e),
                "timestamp": time.time()
            }


def load_processed_samples_from_output(output_file: str) -> set:
    """
    从输出文件中加载已处理的样本ID
    
    Args:
        output_file: 输出文件路径
        
    Returns:
        已处理样本ID的集合
    """
    processed_ids = set()
    if Path(output_file).exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        result = json.loads(line)
                        sample_id = result.get('id')
                        if sample_id:
                            processed_ids.add(sample_id)
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析输出文件第 {line_num} 行失败: {e}")
                        continue
            logger.info(f"从输出文件加载了 {len(processed_ids)} 个已处理的样本ID")
        except Exception as e:
            logger.error(f"读取输出文件失败: {e}")
    else:
        logger.info(f"输出文件不存在: {output_file}")
    return processed_ids


def load_dataset(file_path: str, limit: Optional[int] = None) -> List[Dict]:
    """
    加载数据集
    
    Args:
        file_path: 数据文件路径
        limit: 限制加载的样本数量
        
    Returns:
        数据样本列表
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"解析第 {i+1} 行失败: {e}")
                continue
    
    return samples


def save_results(results: List[Dict], output_file: str, append: bool = False):
    """
    保存结果到文件
    
    Args:
        results: 结果列表
        output_file: 输出文件路径
        append: 是否追加模式（True为追加，False为覆盖）
    """
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = 'a' if append else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    action = "追加到" if append else "保存到"
    logger.info(f"结果已{action}: {output_file}")


def split_into_batches(samples: List[Dict], batch_size: int) -> List[List[Dict]]:
    """
    将样本分割成批次（此函数已弃用，保留用于兼容性）
    
    Args:
        samples: 样本列表
        batch_size: 批次大小
        
    Returns:
        批次列表
    """
    batches = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        batches.append(batch)
    return batches

def calculate_metrics(results: List[Dict]) -> Dict:
    """
    计算评估指标
    
    Args:
        results: 评估结果列表
        
    Returns:
        指标字典
    """
    total_samples = len(results)
    successful_detections = 0
    error_samples = 0
    
    for result in results:
        if "error" in result:
            error_samples += 1
        elif "gemini_detection" in result:
            successful_detections += 1
    
    metrics = {
        "total_samples": total_samples,
        "successful_detections": successful_detections,
        "error_samples": error_samples,
        "success_rate": successful_detections / total_samples if total_samples > 0 else 0
    }
    
    return metrics


def process_sample_worker(sample_queue: Queue, model: str, api_key: str, base_url: str, result_queue: Queue, 
                         progress_queue: Queue, worker_id: int):
    """
    工作进程函数，处理单个样本数据
    
    Args:
        sample_queue: 样本队列
        model: 模型名称
        api_key: API 密钥
        base_url: API 基础 URL
        result_queue: 结果队列
        progress_queue: 进度队列
        worker_id: 工作进程 ID
    """
    try:
        # 为每个进程创建独立的检测器
        detector = GeminiAnomalyDetector(model, api_key, base_url)
        
        while True:
            try:
                # 从队列中获取样本，设置超时避免无限等待
                sample = sample_queue.get(timeout=1)
                if sample is None:  # 结束信号
                    break
                    
                result = detector.evaluate_sample(sample)
                result_queue.put(result)
                
                # 发送进度更新
                progress_queue.put({
                    'worker_id': worker_id,
                    'sample_id': sample.get('id', 'unknown'),
                    'status': 'completed'
                })
                
            except Exception as e:
                if 'sample' in locals() and sample:
                    error_result = {
                        "id": sample.get("id"),
                        "error": str(e),
                        "timestamp": time.time()
                    }
                    result_queue.put(error_result)
                    
                    # 发送错误进度更新
                    progress_queue.put({
                        'worker_id': worker_id,
                        'sample_id': sample.get('id', 'unknown'),
                        'status': 'error',
                        'error': str(e)
                    })
                else:
                    break  # 队列超时，退出循环
            
    except Exception as e:
        logger.error(f"工作进程 {worker_id} 发生错误: {e}")


def process_multiprocessing(samples: List[Dict], model: str, api_key: str, base_url: str,
                          batch_size: int) -> List[Dict]:
    """
    使用多进程处理所有样本
    
    Args:
        samples: 样本列表
        model: 模型名称
        api_key: API 密钥
        base_url: API 基础 URL
        batch_size: 并行进程数
        
    Returns:
        处理结果列表
    """
    logger.info(f"使用 {batch_size} 个并行进程处理 {len(samples)} 个样本")
    
    # 创建队列
    manager = Manager()
    sample_queue = manager.Queue()
    result_queue = manager.Queue()
    progress_queue = manager.Queue()
    
    # 将所有样本放入队列
    for sample in samples:
        sample_queue.put(sample)
    
    # 添加结束信号
    for _ in range(batch_size):
        sample_queue.put(None)
    
    # 创建进程
    processes = []
    for i in range(batch_size):
        p = Process(
            target=process_sample_worker,
            args=(sample_queue, model, api_key, base_url, result_queue, progress_queue, i)
        )
        processes.append(p)
        p.start()
    
    # 收集结果
    all_results = []
    completed_samples = 0
    
    with tqdm(total=len(samples), desc="处理样本") as pbar:
        while completed_samples < len(samples):
            try:
                # 检查是否有新的进度更新
                try:
                    progress_update = progress_queue.get_nowait()
                    if progress_update.get('status') == 'error':
                        logger.warning(f"样本 {progress_update.get('sample_id')} 处理失败: {progress_update.get('error')}")
                    completed_samples += 1
                    pbar.update(1)
                except:
                    pass
                
                # 检查是否有完成的结果
                try:
                    result = result_queue.get_nowait()
                    all_results.append(result)
                except:
                    pass
                
                # 检查进程状态
                for p in processes:
                    if not p.is_alive() and p.exitcode != 0:
                        logger.error(f"进程异常退出，退出码: {p.exitcode}")
                
                time.sleep(0.1)  # 短暂休眠避免过度占用 CPU
                
            except KeyboardInterrupt:
                logger.info("收到中断信号，正在停止所有进程...")
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                        p.join()
                break
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    return all_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用 Gemini 模型进行异常检测（样本级并行处理）")
    parser.add_argument("--input", type=str, 
                       default="data_processing/unified_dataset/final_data/whowhen.jsonl", 
                       help="输入数据文件路径")
    parser.add_argument("--output", type=str, 
                       default="evaluation/whowhen_gemini_results_multiprocessing.jsonl", 
                       help="输出结果文件路径")
    parser.add_argument("--limit", type=int, help="限制处理的样本数量")
    parser.add_argument("--batch_size", type=int, default=4, help="并行进程数")
    parser.add_argument("--api_key", type=str, help="Gemini API 密钥")
    parser.add_argument("--base_url", type=str, 
                       default="https://generativelanguage.googleapis.com/v1beta/openai/",
                       help="API 基础 URL")
    parser.add_argument("--model", type=str,
                        default="qwen/qwen-2.5-72b-instruct:free",
                        help="")
    parser.add_argument("--resume", action="store_true", help="从输出文件加载已处理的样本ID并跳过")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.input).exists():
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 获取 API 密钥
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
    
    # 加载已处理的样本ID
    if args.resume:
        processed_ids = load_processed_samples_from_output(args.output)
    else:
        processed_ids = set()
    
    # 加载数据集，跳过已处理的样本
    logger.info(f"加载数据集: {args.input}")
    samples = load_dataset(args.input, args.limit)
    logger.info(f"加载了 {len(samples)} 个样本")
    
    # 过滤掉已处理的样本
    samples = [s for s in samples if s.get('id') not in processed_ids]
    logger.info(f"过滤掉 {len(processed_ids)} 个已处理的样本，剩余 {len(samples)} 个样本")
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用多进程处理样本
    logger.info(f"使用多进程模式处理，并行进程数: {args.batch_size}")
    results = process_multiprocessing(samples, args.model, api_key, args.base_url, args.batch_size)
    
    # 记录结束时间
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 计算指标
    metrics = calculate_metrics(results)
    logger.info(f"评估完成: {metrics}")
    logger.info(f"总处理时间: {processing_time:.2f} 秒")
    logger.info(f"平均每个样本处理时间: {processing_time/len(samples):.2f} 秒")
    
    # 保存结果
    save_results(results, args.output, append=args.resume)
    
    # 打印详细统计
    print("\n" + "="*60)
    print("评估结果统计")
    print("="*60)
    print(f"总样本数: {metrics['total_samples']}")
    print(f"成功检测: {metrics['successful_detections']}")
    print(f"错误样本: {metrics['error_samples']}")
    print(f"成功率: {metrics['success_rate']:.2%}")
    print(f"总处理时间: {processing_time:.2f} 秒")
    print(f"平均每个样本处理时间: {processing_time/len(samples):.2f} 秒")
    print(f"并行进程数: {args.batch_size}")
    print(f"处理模式: 多进程并行")
    print("="*60)


if __name__ == "__main__":
    # 设置多进程启动方法
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    main() 