#!/usr/bin/env python3
"""
核材料领域专家模型 SFT 训练脚本
基于 Qwen2-7B-Instruct + QLoRA

硬件要求: 2 x RTX A6000 48GB (或单卡 24GB+)
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
# 模型配置
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # 或使用本地路径
OUTPUT_DIR = "./output/nuclear_sft_qwen2_7b"
MAX_seq_LENGTH = 2048

# 训练超参数 (根据A6000 48GB优化)
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_GRAD_NORM = 1.0

# LoRA配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ============== 数据准备 ==============
def prepare_dataset():
    """加载并预处理SFT数据"""
    # 加载JSONL格式的训练数据
    dataset = load_dataset(
        "json",
        data_files="cleaned_data/training_data.jsonl",
        split="train"
    )
    
    print(f"原始数据量: {len(dataset)} 条")
    
    # 数据预处理 - 提取对话格式
    def format_conversations(example):
        # 提取qa_pairs中的对话
        if "qa_pairs" in example and len(example["qa_pairs"]) > 0:
            conversations = example["qa_pairs"][0].get("conversations", [])
            return {"text": conversations}
        return {"text": []}
    
    dataset = dataset.map(format_conversations, remove_columns=dataset.column_names)
    
    # 过滤空数据
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    
    print(f"有效数据量: {len(dataset)} 条")
    return dataset

# ============== 模型加载 ==============
def load_model_and_tokenizer():
    """加载模型和分词器"""
    print(f"正在加载模型: {MODEL_NAME}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 配置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型 (4bit量化)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,  # QLoRA 4bit量化
    )
    
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
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# ============== 训练 ==============
def train():
    """执行SFT训练"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据和模型
    dataset = prepare_dataset()
    model, tokenizer = load_model_and_tokenizer()
    
    # 配置训练参数
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
        bf16=True,  # 使用BF16混合精度
        dataloader_num_workers=4,
        remove_unused_columns=False,
        warmup_ratio=0.1,
        report_to="none",
    )
    
    # 初始化SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_seq_LENGTH,
        packing=False,  # 是否使用序列打包
    )
    
    # 开始训练
    print("\n" + "="*50)
    print("开始SFT训练...")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*50 + "\n")
    
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print("\n训练完成！模型已保存到:", os.path.join(OUTPUT_DIR, "final"))

if __name__ == "__main__":
    train()
