#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核材料DAPT模型评估脚本 V2
使用英文问答，并进行多维度打分
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

# 英文测试问题 - 附带期望答案要点
TEST_QUESTIONS = [
    {
        "question": "What is nuclear material?",
        "keywords": ["radioactive", "uranium", "plutonium", "fuel", "fission"],
        "category": "basic"
    },
    {
        "question": "What are the main components of nuclear fuel?",
        "keywords": ["uranium", "UO2", "pellet", "cladding", "zircaloy"],
        "category": "fuel"
    },
    {
        "question": "What is the basic principle of a nuclear reactor?",
        "keywords": ["fission", "chain reaction", "neutron", "energy", "control"],
        "category": "reactor"
    },
    {
        "question": "How is nuclear waste processed?",
        "keywords": ["storage", "disposal", "reprocessing", "radioactive", "geological"],
        "category": "waste"
    },
    {
        "question": "What are the advantages of nuclear power generation?",
        "keywords": ["low carbon", "efficient", "reliable", "baseload", "emissions"],
        "category": "energy"
    },
    {
        "question": "How to solve the corrosion problem of nuclear materials?",
        "keywords": ["corrosion", "oxidation", "coating", "material selection", "environment"],
        "category": "material"
    },
    {
        "question": "What steps are included in the nuclear fuel cycle?",
        "keywords": ["mining", "enrichment", "fabrication", "burnup", "reprocessing"],
        "category": "fuel"
    },
    {
        "question": "What are the basic principles of nuclear safety?",
        "keywords": ["defense in depth", "containment", "backup", "redundancy", "safety"],
        "category": "safety"
    },
    {
        "question": "What are the mechanical properties of nuclear materials?",
        "keywords": ["strength", "ductility", "radiation damage", "embrittlement", "creep"],
        "category": "material"
    },
    {
        "question": "What is the radiation damage mechanism of nuclear materials?",
        "keywords": ["displacement", "transmutation", "helium", "void", "neutron"],
        "category": "material"
    },
]

# 评分标准
SCORING_CRITERIA = {
    "relevance": {
        "weight": 0.25,
        "description": "Answer relevance to the question"
    },
    "accuracy": {
        "weight": 0.30,
        "description": "Factual accuracy of the content"
    },
    "completeness": {
        "weight": 0.25,
        "description": "Coverage of key points"
    },
    "clarity": {
        "weight": 0.20,
        "description": "Clarity and coherence of explanation"
    }
}

# ========================
# 加载模型
# ========================

def load_model():
    """加载基础模型和训练好的LoRA适配器"""
    print("Loading base model...")
    model, processor = get_model_processor(MODEL_PATH)
    
    print("Loading LoRA adapter...")
    model = Swift.from_pretrained(model, ADAPTER_PATH)
    
    print("Creating template...")
    template = get_template(processor, enable_thinking=False)
    
    print("Creating inference engine...")
    engine = TransformersEngine(model, template=template)
    
    return engine

# ========================
# 评分函数
# ========================

def calculate_keyword_score(answer, keywords):
    """基于关键词匹配计算得分"""
    answer_lower = answer.lower()
    matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return matched / len(keywords) if keywords else 0

def calculate_length_score(answer):
    """基于回答长度计算得分"""
    # 最佳长度范围: 100-500字符
    length = len(answer)
    if length < 50:
        return 0.3
    elif length < 100:
        return 0.6
    elif length <= 500:
        return 1.0
    elif length <= 1000:
        return 0.8
    else:
        return 0.6

def calculate_coherence_score(answer):
    """基于回答连贯性计算得分"""
    # 检查是否为空或过短
    if len(answer) < 20:
        return 0.2
    
    # 清理答案（移除思考标记）
    clean_answer = answer.replace("<think>", "").replace("</think>", "").strip()
    
    # 检查是否包含思考标记
    if "<think>" in answer or "</think>" in answer:
        # 如果答案主要是思考内容，给低分
        if len(clean_answer) < 50:
            return 0.3
        return 0.7
    
    # 检查是否有实质内容
    if len(clean_answer) > 50:
        return 1.0
    
    return 0.5

def score_answer(question_data, answer):
    """对答案进行全面评分"""
    scores = {}
    
    # 1. 关键词匹配得分 (40%)
    keyword_score = calculate_keyword_score(answer, question_data.get("keywords", []))
    scores["keyword_match"] = keyword_score * 100
    
    # 2. 长度得分 (20%)
    length_score = calculate_length_score(answer)
    scores["length"] = length_score * 100
    
    # 3. 连贯性得分 (20%)
    coherence_score = calculate_coherence_score(answer)
    scores["coherence"] = coherence_score * 100
    
    # 4. 实体相关性得分 (20%)
    # 检查是否包含核材料相关术语
    nuclear_terms = ["nuclear", "uranium", "reactor", "fuel", "radiation", "atom", "fission"]
    has_nuclear_content = any(term in answer.lower() for term in nuclear_terms)
    scores["nuclear_relevance"] = 100 if has_nuclear_content else 30
    
    # 计算总分
    total_score = (
        scores["keyword_match"] * 0.4 +
        scores["length"] * 0.2 +
        scores["coherence"] * 0.2 +
        scores["nuclear_relevance"] * 0.2
    )
    scores["total"] = total_score
    
    return scores

# ========================
# 推理测试
# ========================

def test_inference(engine):
    """测试模型推理"""
    print("\n" + "=" * 60)
    print("Starting Model Inference Test (English)")
    print("=" * 60)
    
    results = []
    
    for i, q_data in enumerate(TEST_QUESTIONS, 1):
        question = q_data["question"]
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] Question: {question}")
        
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
        clean_response = ""
        try:
            resp_list = engine.infer([infer_request], request_config=request_config)
            response = resp_list[0].choices[0].message.content
            
            # 清理回答
            clean_response = response.replace("<think>", "").replace("</think>", "").strip()
            
            # 计算得分
            scores = score_answer(q_data, clean_response)
            
            print(f"Answer: {clean_response[:150]}...")
            print(f"Score: {scores['total']:.1f}/100")
            
            results.append({
                "question": question,
                "category": q_data["category"],
                "keywords": q_data["keywords"],
                "answer": clean_response,
                "length": len(clean_response),
                "scores": scores
            })
            
        except Exception as e:
            print(f"Inference failed: {e}")
            results.append({
                "question": question,
                "category": q_data["category"],
                "keywords": q_data["keywords"],
                "answer": f"ERROR: {str(e)}",
                "length": 0,
                "scores": {
                    "keyword_match": 0,
                    "length": 0,
                    "coherence": 0,
                    "nuclear_relevance": 0,
                    "total": 0
                }
            })
    
    return results

# ========================
# 评估指标
# ========================

def calculate_metrics(results):
    """计算评估指标"""
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    
    # 分类别统计
    categories = {}
    total_score = 0
    valid_count = 0
    
    for result in results:
        if result["scores"]["total"] > 0:
            total_score += result["scores"]["total"]
            valid_count += 1
            
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"scores": [], "count": 0}
            categories[cat]["scores"].append(result["scores"]["total"])
            categories[cat]["count"] += 1
    
    avg_score = total_score / valid_count if valid_count > 0 else 0
    
    # 打印各类别得分
    print(f"\nTotal questions: {len(results)}")
    print(f"Valid responses: {valid_count}")
    print(f"Success rate: {valid_count/len(results)*100:.1f}%")
    print(f"\nOverall Average Score: {avg_score:.1f}/100")
    
    print("\n--- Scores by Category ---")
    for cat, data in categories.items():
        cat_avg = sum(data["scores"]) / len(data["scores"])
        print(f"{cat}: {cat_avg:.1f}/100 ({data['count']} questions)")
    
    # 详细分数统计
    print("\n--- Detailed Score Breakdown ---")
    score_components = {
        "keyword_match": [],
        "length": [],
        "coherence": [],
        "nuclear_relevance": []
    }
    
    for result in results:
        for key in score_components.keys():
            if result["scores"][key] > 0:
                score_components[key].append(result["scores"][key])
    
    for key, scores in score_components.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"{key}: {avg:.1f}/100")
    
    return {
        "total_questions": len(results),
        "valid_responses": valid_count,
        "success_rate": valid_count/len(results)*100,
        "average_score": avg_score,
        "categories": {cat: {"avg": sum(data["scores"])/len(data["scores"]), "count": data["count"]} 
                      for cat, data in categories.items()},
        "score_components": {key: sum(scores)/len(scores) if scores else 0 
                            for key, scores in score_components.items()}
    }

# ========================
# 保存结果
# ========================

def save_results(results, metrics):
    """保存评估结果"""
    output_dir = "output/evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    results_file = os.path.join(output_dir, "dapt_evaluation_v2_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_PATH,
            "adapter": ADAPTER_PATH,
            "results": results,
            "metrics": metrics
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # 保存摘要
    summary_file = os.path.join(output_dir, "dapt_evaluation_v2_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Nuclear Material DAPT Model Evaluation Summary (V2)\n")
        f.write("English QA with Scoring\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Base Model: {MODEL_PATH}\n")
        f.write(f"Adapter Path: {ADAPTER_PATH}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Overall Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Questions: {metrics['total_questions']}\n")
        f.write(f"Valid Responses: {metrics['valid_responses']}\n")
        f.write(f"Success Rate: {metrics['success_rate']:.1f}%\n")
        f.write(f"Average Score: {metrics['average_score']:.1f}/100\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Score Breakdown\n")
        f.write("=" * 60 + "\n")
        for key, score in metrics['score_components'].items():
            f.write(f"{key}: {score:.1f}/100\n")
        f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("Scores by Category\n")
        f.write("=" * 60 + "\n")
        for cat, data in metrics['categories'].items():
            f.write(f"{cat}: {data['avg']:.1f}/100 ({data['count']} questions)\n")
        f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("Detailed Results\n")
        f.write("=" * 60 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"[{i}] Question: {result['question']}\n")
            f.write(f"    Category: {result['category']}\n")
            f.write(f"    Score: {result['scores']['total']:.1f}/100\n")
            f.write(f"    - Keyword Match: {result['scores']['keyword_match']:.1f}\n")
            f.write(f"    - Length: {result['scores']['length']:.1f}\n")
            f.write(f"    - Coherence: {result['scores']['coherence']:.1f}\n")
            f.write(f"    - Nuclear Relevance: {result['scores']['nuclear_relevance']:.1f}\n")
            f.write(f"    Answer: {result['answer'][:200]}...\n")
            f.write("-" * 60 + "\n\n")
    
    print(f"Summary saved to: {summary_file}")

# ========================
# 主函数
# ========================

def main():
    print("Nuclear Material DAPT Model Evaluation V2")
    print("English QA with Multi-dimensional Scoring")
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
    print("Evaluation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
