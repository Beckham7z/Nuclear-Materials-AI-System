#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核材料领域 DAPT 训练 - 使用 Swift 框架
Qwen3.5-2B (继续预训练)

Swift框架优势：
- 配置简单
- 自动处理数据tokenization
- 支持继续预训练
"""

import os
import torch

# 设置可见GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift import Swift, LoRAConfig, InferEngine, InferRequest
from swift.trainers import Trainer
from modelscope import Model, Dataset

# ========================
# 基本路径配置
# ========================
MODEL_PATH = "Qwen/Qwen3.5-2B"  # ModelScope模型ID
OUTPUT_DIR = "./output/nuclear_sft_swift"
DATA_PATH = "dapt_sft_data/dapt_training_data.txt"  # 纯文本数据

# ========================
# 训练参数配置
# ========================
# LoRA配置
LORA_CONFIG = {
    'r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.1,
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                       'up_proj', 'down_proj', 'gate_proj'],
    'bias': 'none',
    'task_type': 'CAUSAL_LM',
}

# 训练参数
TRAINING_ARGS = {
    'learning_rate': 1e-5,  # DAPT用较低学习率
    'num_train_epochs': 3,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,
    'max_grad_norm': 0.3,
    'warmup_steps': 50,
    'weight_decay': 0.01,
    'lr_scheduler_type': 'cosine',
    'logging_steps': 1,
    'save_strategy': 'epoch',
    'save_total_limit': 2,
    'bf16': True,
    'max_seq_length': 2048,
    'output_dir': OUTPUT_DIR,
}

# ========================
# 数据准备
# ========================

def prepare_dataset():
    """加载纯文本数据集"""
    # Swift支持直接加载本地文本文件
    # 会自动进行tokenization
    dataset = Dataset.from_json(DATA_PATH)
    print(f"数据集大小: {len(dataset)}")
    return dataset


# ========================
# 训练主流程
# ========================

def train():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载模型
    print("加载模型...")
    model = Model.from_pretrained(
        MODEL_PATH,
        model_type='qwen2.5',
        dtype='bf16',
    )
    
    # 2. 配置LoRA
    print("配置LoRA...")
    lora_config = LoRAConfig(**LORA_CONFIG)
    model = Swift.from_pretrained(model, lora_config)
    
    # 3. 加载数据
    print("加载数据集...")
    train_dataset = prepare_dataset()
    
    # 4. 配置训练参数
    print("配置训练参数...")
    training_args = {
        **TRAINING_ARGS,
        'train_dataset': train_dataset,
        'model': model,
    }
    
    # 5. 创建训练器
    print("创建训练器...")
    trainer = Trainer(args=training_args)
    
    # 6. 开始训练
    print("\n" + "=" * 60)
    print("开始 DAPT 训练 (Swift框架)")
    print("配置要点:")
    print(f"  - 模型: {MODEL_PATH}")
    print(f"  - 数据: {DATA_PATH}")
    print(f"  - 学习率: {TRAINING_ARGS['learning_rate']}")
    print(f"  - 训练轮数: {TRAINING_ARGS['num_train_epochs']}")
    print(f"  - LoRA r: {LORA_CONFIG['r']}")
    print(f"  - max_seq_length: {TRAINING_ARGS['max_seq_length']}")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # 7. 保存模型
    print("保存模型...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print(f"模型已保存到: {os.path.join(OUTPUT_DIR, 'final')}")


# ========================
# 推理示例
# ========================

def infer():
    """推理示例 - 验证模型效果"""
    print("\n加载推理引擎...")
    
    # 加载微调后的模型
    model = Model.from_pretrained(
        os.path.join(OUTPUT_DIR, "final"),
        model_type='qwen2.5',
    )
    lora_config = LoRAConfig(**LORA_CONFIG)
    model = Swift.from_pretrained(model, lora_config)
    
    # 创建推理引擎
    infer_engine = InferEngine(model)
    
    # 测试问题
    test_questions = [
        "什么是核材料？",
        "核燃料的主要成分是什么？",
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        
        # 创建推理请求
        infer_request = InferRequest(
            messages=[{
                "role": "user",
                "content": question
            }]
        )
        
        # 推理
        resp = infer_engine.infer([infer_request])
        print(f"回答: {resp[0].choices[0].message.content}")


if __name__ == "__main__":
    # 训练
    train()
    
    # 训练完成后可以运行推理测试
    # infer()
