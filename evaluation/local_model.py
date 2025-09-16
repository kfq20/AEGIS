#!/usr/bin/env python3
"""
Qwen Anomaly Detection Script
使用 Qwen 模型对 whowhen.jsonl 数据集进行异常检测评测

Usage:
    python qwen_anomaly_detection.py --input /home/fanqi/verl/data/maserror/unified_dataset/whowhen.jsonl --output AMEeval/results_qwen.jsonl
    python qwen_anomaly_detection.py --input /home/fanqi/verl/data/maserror/unified_dataset/whowhen.jsonl --output AMEeval/results_qwen.jsonl --limit 10
"""

import json
import argparse
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qwen_anomaly_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class QwenAnomalyDetector:
    """使用 Qwen 模型进行异常检测的类"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", lora_adapter_path: Optional[str] = None, max_new_tokens: int = 256, enable_thinking: bool = False):
        """
        初始化 Qwen 模型
        
        Args:
            model_name: Qwen 模型名称或路径
            lora_adapter_path: LoRA适配器路径（可选）
            max_new_tokens: 生成的最大新token数量
            enable_thinking: 是否启用思考模式（用于 Qwen3 特殊处理）
        """
        self.model_name = model_name
        self.enable_thinking = enable_thinking  # 使用参数传入的值
        self.max_new_tokens = max_new_tokens  # 添加 max_new_tokens 属性
        logger.info(f"正在加载模型: {model_name}")
        logger.info(f"\n========== model_name: {self.model_name}==========\n")
        
        # 检查是否为LoRA checkpoint路径
        model_path = Path(model_name)
        if model_path.exists() and (model_path / "lora_adapter").exists():
            # 这是一个包含LoRA的checkpoint路径
            lora_adapter_path = str(model_path / "lora_adapter")
            # 从adapter_config.json中读取base model
            try:
                import json
                with open(model_path / "lora_adapter" / "adapter_config.json", 'r') as f:
                    adapter_config = json.load(f)
                    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
                logger.info(f"检测到LoRA checkpoint，使用base model: {base_model_name}")
                logger.info(f"LoRA adapter路径: {lora_adapter_path}")
            except Exception as e:
                logger.warning(f"无法读取adapter_config.json: {e}，使用默认base model")
                base_model_name = "Qwen/Qwen2.5-7B-Instruct"
        else:
            base_model_name = model_name
        
        # 加载base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # 如果有LoRA adapter，则加载它
        if lora_adapter_path:
            logger.info(f"正在加载LoRA adapter: {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
            logger.info("LoRA adapter加载完成")
        
        # 清理不必要的GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("模型加载完成")
        
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
            # agent_role = entry.get('agent_role', '')
            content = entry.get('content', '')
            # phase = entry.get('phase', '')
            
            conversation_text += f"Step {step} - {agent_name}:\n{content}\n\n"
        
        return conversation_text.strip()[:100000]
    
    def filter_long_samples(self, conversation_texts: List[str], max_length: int = 8192) -> tuple[List[str], List[int], Dict]:
        """
        不再过滤样本，直接返回所有样本
        
        Args:
            conversation_texts: 对话文本列表
            max_length: 最大token长度（保留参数以兼容调用，但不使用）
            
        Returns:
            (所有对话文本列表, 所有样本索引列表, 统计信息字典)
        """
        if not conversation_texts:
            return [], [], {"total": 0, "filtered": 0, "kept": 0}
            
        # 不再进行任何过滤，直接返回所有样本
        kept_indices = list(range(len(conversation_texts)))
        stats = {"total": len(conversation_texts), "filtered": 0, "kept": len(conversation_texts)}
        
        logger.info(f"数据统计: 总计 {stats['total']} 个样本, 全部保留 {stats['kept']} 个, 过滤 {stats['filtered']} 个")
        return conversation_texts, kept_indices, stats

    def detect_anomalies_batch(self, conversation_texts: List[str], max_retries: int = 3) -> List[Optional[Dict]]:
        """
        批量使用 Qwen 模型检测异常
        
        Args:
            conversation_texts: 对话文本列表
            max_retries: 最大重试次数
            
        Returns:
            检测结果字典列表
        """
        if not conversation_texts:
            return []
            
        prompt_template = self.load_prompt_template('/home/fanqi/verl/AMEeval/prompt.txt')
        
        # 为每个对话构建prompt
        prompts = [prompt_template.format(conversation_text=text) for text in conversation_texts]
        
        for attempt in range(max_retries):
            try:
                # 构建批量消息格式
                batch_messages = []
                for prompt in prompts:
                    messages = [
                        {"role": "system", "content": "You are a precise analyst. Please follow the instructions carefully and provide detailed reasoning before giving your final JSON answer."},
                        {"role": "user", "content": prompt}
                    ]
                    batch_messages.append(messages)
                
                # 批量应用聊天模板并直接tokenize，避免截断问题 (添加 qwen3 特殊处理)
                batch_texts = []
                for messages in batch_messages:
                    if "qwen3" in self.model_name.lower():
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=self.enable_thinking
                        )
                    else:
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    batch_texts.append(text)
                
                # 批量编码输入
                model_inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=False  # 不再截断输入
                ).to(self.model.device)
                
                
                # 批量生成回复
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=self.max_new_tokens,
                        # temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                batch_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                batch_results = []
                batch_raw_response = []
                for response in batch_response:
                    try:
                        batch_raw_response.append(response)
                        result = self._parse_response(response)
                        batch_results.append(result)
                    except Exception as e:
                        logger.warning(f"解析批量结果失败: {e}")
                        batch_results.append({"faulty_agents": []})
                
                # 清理GPU内存
                del model_inputs, generated_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return batch_results, batch_raw_response
                    
            except Exception as e:
                logger.error(f"批量推理失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return [{"faulty_agents": []} for _ in conversation_texts], batch_raw_response
                time.sleep(2)
        
        return [{"faulty_agents": []} for _ in conversation_texts], batch_raw_response
    
    def _parse_response(self, response_text: str) -> Dict:
        """
        解析模型响应为JSON格式
        
        Args:
            response_text: 模型响应文本
            
        Returns:
            解析后的结果字典
        """
        # 清理响应文本，移除可能的 markdown 格式
        clean_response = response_text.strip()
        
        # 如果响应包含 markdown 代码块，移除它们
        if '```json' in clean_response:
            import re
            json_match = re.search(r'```json\s*({.*?})\s*```', clean_response, re.DOTALL)
            if json_match:
                clean_response = json_match.group(1)
        elif '```' in clean_response:
            # 移除普通代码块
            clean_response = clean_response.replace('```json', '').replace('```', '').strip()
        
        # 尝试从响应中提取JSON对象
        # 查找最后一个完整的JSON对象
        import re
        json_pattern = r'{\s*"faulty_agents"\s*:\s*\[.*?\]\s*}'
        json_matches = re.findall(json_pattern, clean_response, re.DOTALL)
        
        if json_matches:
            # 使用最后一个匹配的JSON对象（通常是最终答案）
            json_str = json_matches[-1]
            result = json.loads(json_str)
            return result
        else:
            # 如果没有找到标准格式，尝试解析整个响应作为JSON
            result = json.loads(clean_response)
            return result
    
    def evaluate_samples_batch(self, samples: List[Dict]) -> List[Dict]:
        """
        批量评估样本，会先过滤超长样本
        
        Args:
            samples: 数据样本列表
            
        Returns:
            评估结果列表（包含被过滤样本的错误信息）
        """
        if not samples:
            return []
            
        try:
            # 批量提取对话文本
            conversation_texts = []
            for sample in samples:
                input_data = sample.get('input', {})
                conversation_text = self.extract_conversation_text(input_data)
                conversation_texts.append(conversation_text)
            
            # 不再过滤样本，直接处理所有样本
            filtered_texts, kept_indices, filter_stats = self.filter_long_samples(conversation_texts, max_length=8192)
            
            # 批量检测异常（处理所有样本）
            detection_results, batch_raw_response = self.detect_anomalies_batch(filtered_texts)
            
            # 为所有样本构建结果
            results = []
            for i, (detection_result, raw_response) in enumerate(zip(detection_results, batch_raw_response)):
                sample = samples[i]
                result = {
                    "id": sample.get("id"),
                    "metadata": sample.get("metadata"),
                    "input": sample.get("input"),
                    "ground_truth": sample.get("output"),
                    "model_detection": detection_result,
                    "raw_response": raw_response,
                    "original_sample": sample,
                    "filter_stats": filter_stats
                }
                print(f"raw_response: {raw_response}")
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"批量评估失败: {e}")
            # 如果批量处理失败，返回错误结果
            error_results = []
            for sample in samples:
                error_result = {
                    "id": sample.get("id"),
                    "error": str(e),
                    "original_sample": sample,
                    "timestamp": time.time()
                }
                error_results.append(error_result)
            return error_results


def create_sample_hash(sample: Dict) -> str:
    """
    为样本创建唯一哈希标识符
    
    Args:
        sample: 样本数据
        
    Returns:
        样本的哈希值
    """
    # 创建一个包含关键字段的字符串来生成哈希
    # 使用id、query、conversation_history来确保唯一性
    input_data = sample.get('input', {})
    hash_data = {
        'id': sample.get('id', ''),
        'query': input_data.get('query', ''),
        'conversation_history': input_data.get('conversation_history', [])
    }
    
    # 将数据转换为JSON字符串并生成MD5哈希
    hash_string = json.dumps(hash_data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(hash_string.encode('utf-8')).hexdigest()


def load_existing_results(output_file: str) -> tuple[List[Dict], set]:
    """
    加载现有的结果文件，返回已有结果和已处理的样本哈希集合
    
    Args:
        output_file: 输出结果文件路径
        
    Returns:
        tuple: (现有结果列表, 已处理样本哈希集合)
    """
    existing_results = []
    processed_hashes = set()
    
    if Path(output_file).exists():
        logger.info(f"发现现有结果文件: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line.strip())
                    existing_results.append(result)
                    # 为已处理的样本创建哈希并添加到集合中
                    if 'original_sample' in result:
                        sample_hash = create_sample_hash(result['original_sample'])
                        processed_hashes.add(sample_hash)
                    # 为了兼容旧格式，如果没有original_sample但有其他字段，也尝试创建哈希
                    elif 'id' in result:
                        # 从结果中重构样本数据来计算哈希，保持与真实样本相同的结构
                        input_data = result.get('input', {})
                        reconstructed_sample = {
                            'id': result.get('id', ''),
                            'input': {
                                'query': input_data.get('query', ''),
                                'conversation_history': input_data.get('conversation_history', [])
                            }
                        }
                        sample_hash = create_sample_hash(reconstructed_sample)
                        processed_hashes.add(sample_hash)
                except json.JSONDecodeError as e:
                    logger.warning(f"解析现有结果第 {line_num} 行失败: {e}")
                    continue
        
        logger.info(f"加载了 {len(existing_results)} 个现有结果，已处理样本哈希数: {len(processed_hashes)}")
    else:
        logger.info(f"输出文件不存在，将从头开始处理")
    
    return existing_results, processed_hashes


def filter_unprocessed_samples(samples: List[Dict], processed_hashes: set) -> List[Dict]:
    """
    过滤出未处理的样本
    
    Args:
        samples: 所有样本列表
        processed_hashes: 已处理的样本哈希集合
        
    Returns:
        未处理的样本列表
    """
    unprocessed_samples = []
    for sample in samples:
        sample_hash = create_sample_hash(sample)
        if sample_hash not in processed_hashes:
            unprocessed_samples.append(sample)
    
    logger.info(f"总样本数: {len(samples)}, 已处理: {len(processed_hashes)}, 待处理: {len(unprocessed_samples)}")
    return unprocessed_samples


def append_results(results: List[Dict], output_file: str):
    """
    追加结果到文件（支持断点重续）
    
    Args:
        results: 新结果列表
        output_file: 输出文件路径
    """
    if not results:
        return
        
    with open(output_file, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"追加了 {len(results)} 个结果到: {output_file}")


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


def save_results(results: List[Dict], output_file: str):
    """
    保存结果到文件
    
    Args:
        results: 结果列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"结果已保存到: {output_file}")


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
        elif "model_detection" in result:
            successful_detections += 1
    
    metrics = {
        "total_samples": total_samples,
        "successful_detections": successful_detections,
        "error_samples": error_samples,
        "success_rate": successful_detections / total_samples if total_samples > 0 else 0
    }
    
    return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用 Qwen 模型进行异常检测")
    parser.add_argument("--input", type=str, 
                       default="data/maserror/unified_dataset/test.jsonl", 
                       help="输入数据文件路径")
    parser.add_argument("--output", type=str, 
                       default="AMEeval/ours_qwen25_7b_cot.jsonl", 
                       help="输出结果文件路径")
    parser.add_argument("--limit", type=int, help="限制处理的样本数量")
    parser.add_argument("--model_name", type=str, 
                       default="Qwen/Qwen2.5-14B-Instruct",
                       help="Qwen 模型名称")
    parser.add_argument("--resume", action="store_true", 
                       help="从现有结果文件恢复，跳过已处理的样本")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批量推理的batch size，根据GPU内存调整")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="生成的最大新token数量")
    parser.add_argument("--enable_thinking", action="store_true",
                       help="启用思考模式（用于 Qwen3 模型）")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.input).exists():
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 检查输出目录
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果启用resume模式，先加载现有结果
    existing_results = []
    processed_hashes = set()
    if args.resume:
        existing_results, processed_hashes = load_existing_results(args.output)
    
    # 初始化检测器
    detector = QwenAnomalyDetector(
        args.model_name, 
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking
    )
    
    # 加载数据集
    logger.info(f"加载数据集: {args.input}")
    samples = load_dataset(args.input, args.limit)
    logger.info(f"加载了 {len(samples)} 个样本")
    
    # 如果启用resume模式，过滤已处理的样本
    if args.resume:
        samples = filter_unprocessed_samples(samples, processed_hashes)
        if not samples:
            logger.info("所有样本都已处理完成！")
            return
    
    # 评估样本 - 使用批量处理提升速度
    logger.info("开始异常检测...")
    
    # 设置批量处理的batch_size
    inference_batch_size = args.batch_size  # 从命令行参数获取
    save_batch_size = 20     # 保存批量大小，每处理多少个样本保存一次
    
    all_results = []
    
    # 计算总批次数
    total_batches = (len(samples) - 1) // inference_batch_size + 1
    logger.info(f"总共将处理 {len(samples)} 个样本，分 {total_batches} 个批次，每批次 {inference_batch_size} 个样本")
    
    # 按批量处理样本
    for batch_start in tqdm(range(0, len(samples), inference_batch_size), desc="批量处理", total=total_batches):
        batch_end = min(batch_start + inference_batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]
        
        logger.info(f"处理批次 {batch_start//inference_batch_size + 1}/{(len(samples)-1)//inference_batch_size + 1}: 样本 {batch_start+1}-{batch_end}")
        
        # 批量推理
        batch_results = detector.evaluate_samples_batch(batch_samples)
        all_results.extend(batch_results)
        
        # 定期保存结果，避免内存积累和数据丢失
        if len(all_results) >= save_batch_size or batch_end == len(samples):
            if args.resume:
                append_results(all_results, args.output)
            else:
                # 非resume模式，第一次覆盖文件，之后追加
                if batch_start == 0:
                    save_results(all_results, args.output)
                else:
                    append_results(all_results, args.output)
            
            logger.info(f"已保存 {len(all_results)} 个结果")
            all_results = []  # 清空结果列表
        
        # 添加小延迟避免过度占用资源
        time.sleep(0.5)
    
    # 计算指标和打印统计
    logger.info("处理完成！")
    
    # 加载所有结果并计算最终统计
    if Path(args.output).exists():
        logger.info("正在计算最终统计信息...")
        all_final_results = []
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    all_final_results.append(result)
                except json.JSONDecodeError:
                    continue
        
        final_metrics = calculate_metrics(all_final_results)
        
        logger.info("=" * 60)
        logger.info("最终统计报告:")
        logger.info(f"总样本数: {final_metrics['total_samples']}")
        logger.info(f"成功处理: {final_metrics['successful_detections']}")
        logger.info(f"错误样本: {final_metrics['error_samples']}")
        logger.info(f"成功率: {final_metrics['success_rate']:.2%}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main() 