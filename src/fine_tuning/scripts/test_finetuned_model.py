#!/usr/bin/env python3
"""测试微调后的核材料领域模型"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 配置路径
BASE_MODEL_PATH = "/home/zyx/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B"
FINETUNED_MODEL_PATH = "./output/nuclear_sft_qwen3.5_9b_v2/final"

# 测试问题
TEST_QUESTIONS = [
    # 核材料基础知识
    "什么是核材料？",
    "铀的主要用途是什么？",
    "核燃料是如何工作的？",
    
    # 核材料领域专业问题
    "Hastelloy N合金在熔盐中的腐蚀机理是什么？",
    "什么是四面体形变(Tetrahedral deformation)？",
    "核石墨的辐照行为有哪些特点？",
    
    # 反应堆相关
    "熔盐堆(Molten Salt Reactor)有哪些优势？",
    "核燃料循环分为哪些阶段？",
]

def load_base_model():
    """加载基础模型"""
    print("加载基础模型...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        quantization_config=cfg,
        device_map="cuda"
    )
    
    return model, tok

def load_finetuned_model():
    """加载微调后的模型"""
    print("加载微调模型...")
    tok = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        quantization_config=cfg,
        device_map="cuda"
    )
    
    model = PeftModel.from_pretrained(
        base_model,
        FINETUNED_MODEL_PATH,
        device_map="cuda"
    )
    
    return model, tok

def generate_response(model, tokenizer, question, max_length=512):
    """生成回答"""
    # 构建提示词
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        # 使用贪婪解码避免采样问题
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=1.0,  # 使用固定温度
            top_p=1.0,  # 不使用top-p采样
            do_sample=False,  # 使用贪婪解码
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # 添加重复惩罚
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 assistant 的回答
    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    
    return response.strip()

def compare_models():
    """对比基础模型和微调模型"""
    print("="*60)
    print("模型对比测试")
    print("="*60)
    
    # 加载模型
    base_model, base_tok = load_base_model()
    finetuned_model, finetuned_tok = load_finetuned_model()
    
    print("\n" + "="*60)
    print("开始测试...")
    print("="*60)
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"问题 {i}: {question}")
        print("="*60)
        
        # 基础模型回答
        print("\n[基础模型 Qwen3.5-9B]")
        base_response = generate_response(base_model, base_tok, question)
        print(base_response[:500] + "..." if len(base_response) > 500 else base_response)
        
        # 微调模型回答
        print("\n[微调模型]")
        finetuned_response = generate_response(finetuned_model, finetuned_tok, question)
        print(finetuned_response[:500] + "..." if len(finetuned_response) > 500 else finetuned_response)
        
        print("\n" + "-"*60)

def test_single_model():
    """仅测试微调模型"""
    print("="*60)
    print("测试微调后的核材料专家模型")
    print("="*60)
    
    model, tok = load_finetuned_model()
    
    print("\n模型已加载，可以开始提问！\n")
    
    while True:
        question = input("请输入问题 (输入 q 退出): ").strip()
        if question.lower() in ['q', 'quit', 'exit']:
            break
        
        if not question:
            continue
        
        response = generate_response(model, tok, question)
        print("\n回答:")
        print(response)
        print("\n" + "-"*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_models()
    else:
        test_single_model()
