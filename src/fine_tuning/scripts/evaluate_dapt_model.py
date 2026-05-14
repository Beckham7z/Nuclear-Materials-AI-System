#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核材料DAPT模型评估脚本
评估训练后的模型效果
"""

import os
import json
import torch
from swift import Swift, get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

# ========================
# 配置参数
# ========================
MODEL_PATH = "Qwen/Qwen3.5-2B"  # 基础模型
ADAPTER_PATH = "output/nuclear_sft_dapt_swift_fast/v0-20260310-114209/checkpoint-1890"  # 训练好的LoRA适配器

# 测试问题
TEST_QUESTIONS = [
    "什么是核材料？",
    "核燃料的主要成分是什么？",
    "核反应堆的基本原理是什么？",
    "核废料如何处理？",
    "核能发电的优势有哪些？",
    "核材料的腐蚀问题如何解决？",
    "核燃料循环包括哪些步骤？",
    "核安全的基本原则是什么？",
    "核材料的力学性能有哪些特点？",
    "核材料的辐照损伤机制是什么？",
]

# ========================
# 加载模型
# ========================

def load_model():
    """加载基础模型和训练好的LoRA适配器"""
    print("加载基础模型...")
    model, processor = get_model_processor(MODEL_PATH)
    
    print("加载LoRA适配器...")
    model = Swift.from_pretrained(model, ADAPTER_PATH)
    
    print("创建模板...")
    template = get_template(processor, enable_thinking=False)
    
    print("创建推理引擎...")
    engine = TransformersEngine(model, template=template)
    
    return engine

# ========================
# 推理测试
# ========================

def test_inference(engine):
    """测试模型推理"""
    print("\n" + "=" * 60)
    print("开始模型推理测试")
    print("=" * 60)
    
    results = []
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] 问题: {question}")
        
        # 创建推理请求
        infer_request = InferRequest(
            messages=[{
                "role": "user",
                "content": question
            }]
        )
        
        # 配置请求参数
        request_config = RequestConfig(
            max_tokens=512,
            temperature=0.1,
            stream=False
        )
        
        # 推理
        try:
            resp_list = engine.infer([infer_request], request_config=request_config)
            response = resp_list[0].choices[0].message.content
            
            print(f"回答: {response[:200]}...")
            
            results.append({
                "question": question,
                "answer": response,
                "length": len(response)
            })
            
        except Exception as e:
            print(f"推理失败: {e}")
            results.append({
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "length": 0
            })
    
    return results

# ========================
# 评估指标
# ========================

def calculate_metrics(results):
    """计算评估指标"""
    print("\n" + "=" * 60)
    print("评估指标")
    print("=" * 60)
    
    total_length = 0
    valid_responses = 0
    
    for result in results:
        if not result["answer"].startswith("ERROR"):
            valid_responses += 1
            total_length += result["length"]
    
    avg_length = total_length / valid_responses if valid_responses > 0 else 0
    
    print(f"总问题数: {len(results)}")
    print(f"有效回答数: {valid_responses}")
    print(f"成功率: {valid_responses/len(results)*100:.1f}%")
    print(f"平均回答长度: {avg_length:.0f} 字符")
    
    return {
        "total_questions": len(results),
        "valid_responses": valid_responses,
        "success_rate": valid_responses/len(results)*100,
        "avg_length": avg_length
    }

# ========================
# 保存结果
# ========================

def save_results(results, metrics):
    """保存评估结果"""
    output_dir = "output/evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    results_file = os.path.join(output_dir, "dapt_evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_PATH,
            "adapter": ADAPTER_PATH,
            "results": results,
            "metrics": metrics
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {results_file}")
    
    # 保存摘要
    summary_file = os.path.join(output_dir, "dapt_evaluation_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("核材料DAPT模型评估摘要\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"基础模型: {MODEL_PATH}\n")
        f.write(f"适配器路径: {ADAPTER_PATH}\n\n")
        f.write(f"总问题数: {metrics['total_questions']}\n")
        f.write(f"有效回答数: {metrics['valid_responses']}\n")
        f.write(f"成功率: {metrics['success_rate']:.1f}%\n")
        f.write(f"平均回答长度: {metrics['avg_length']:.0f} 字符\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("示例回答\n")
        f.write("=" * 60 + "\n\n")
        
        for i, result in enumerate(results[:3], 1):
            f.write(f"问题 {i}: {result['question']}\n")
            f.write(f"回答: {result['answer'][:300]}...\n")
            f.write("-" * 60 + "\n\n")
    
    print(f"评估摘要已保存到: {summary_file}")

# ========================
# 主函数
# ========================

def main():
    print("核材料DAPT模型评估")
    print("=" * 60)
    
    # 加载模型
    engine = load_model()
    
    # 推理测试
    results = test_inference(engine)
    
    # 计算指标
    metrics = calculate_metrics(results)
    
    # 保存结果
    save_results(results, metrics)
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
