#!/usr/bin/env python3
"""
核材料领域专家模型 SFT 训练脚本
基于 Ollama qwen3:8b 模型

使用 llama.cpp 将 Ollama 模型转换为 GGUF 格式后进行微调
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# ============== 配置参数 ==============
# 模型配置 - 使用本地 Ollama 模型
MODEL_NAME = "qwen3:8b"  # Ollama 模型名
MODEL_PATH = os.path.expanduser("~/.ollama/models/blobs/")  # Ollama 模型存储路径
OUTPUT_DIR = "./output/nuclear_sft_qwen3_8b"
MAX_SEQ_LENGTH = 2048

# 训练超参数
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_GRAD_NORM = 1.0

# LoRA配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

def find_ollama_model():
    """查找 Ollama qwen3:8b 模型文件"""
    # Ollama 模型通常存储在 ~/.ollama/models/
    ollama_dir = os.path.expanduser("~/.ollama/models/")
    
    if not os.path.exists(ollama_dir):
        print(f"警告: Ollama 模型目录不存在: {ollama_dir}")
        return None
    
    # 列出 blobs 目录中的文件
    blobs_dir = os.path.join(ollama_dir, "blobs")
    if os.path.exists(blobs_dir):
        files = os.listdir(blobs_dir)
        print(f"Ollama blobs 目录中的文件: {files[:5]}...")  # 只显示前5个
    
    return blobs_dir

def prepare_dataset():
    """加载并预处理SFT数据"""
    dataset = load_dataset(
        "json",
        data_files="cleaned_data/training_data.jsonl",
        split="train"
    )
    
    print(f"原始数据量: {len(dataset)} 条")
    
    def format_conversations(example):
        if "qa_pairs" in example and len(example["qa_pairs"]) > 0:
            conversations = example["qa_pairs"][0].get("conversations", [])
            return {"text": conversations}
        return {"text": []}
    
    dataset = dataset.map(format_conversations, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    
    print(f"有效数据量: {len(dataset)} 条")
    return dataset

def load_model_and_tokenizer():
    """加载模型和分词器
    
    由于 Ollama 模型格式特殊，我们尝试从 ModelScope 下载的模型开始
    或者使用 Ollama 的模型文件
    """
    # 首先检查是否有本地模型
    local_model_path = "./models/qwen/Qwen2-7B-Instruct"
    
    if os.path.exists(local_model_path):
        print(f"使用本地模型: {local_model_path}")
        model_path = local_model_path
    else:
        # 尝试使用 Ollama 模型
        print("检查 Ollama 模型...")
        ollama_path = find_ollama_model()
        
        # 临时使用 Qwen2.5 1.5b 模型进行测试
        print("警告: 将使用 qwen2.5:1.5b 模型进行演示")
        model_path = None
    
    # 加载分词器
    if model_path and os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型 (使用4bit量化)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
        )
    else:
        print("错误: 未找到可用的本地模型")
        print("请先下载模型或确保 Ollama 模型可用")
        return None, None
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        inference_mode=False
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train():
    """执行SFT训练"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    dataset = prepare_dataset()
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None:
        print("模型加载失败，训练终止")
        return
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        warmup_ratio=0.1,
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )
    
    print("\n" + "="*50)
    print("开始SFT训练...")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*50 + "\n")
    
    trainer.train()
    
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print("\n训练完成！模型已保存到:", os.path.join(OUTPUT_DIR, "final"))

if __name__ == "__main__":
    train()
