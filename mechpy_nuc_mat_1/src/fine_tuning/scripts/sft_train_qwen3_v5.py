#!/usr/bin/env python3
"""核材料领域SFT训练 - Qwen3.5-2B (修复版 v4)
关键修复:
1. 移除4-bit量化，改用bf16/fp16 (Qwen3.5不推荐4-bit训练)
2. 使用tokenizer.apply_chat_template正确处理对话格式
3. 降低学习率，增加预热步数
4. 修复LoRA target modules，添加MLP层
5. 使用正确的数据格式处理
"""
import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

MODEL_PATH = "/home/zyx/.cache/modelscope/hub/models/Qwen/Qwen3.5-2B"
OUTPUT_DIR = "./output/nuclear_sft_qwen3.5_2b_v5"

def prepare_dataset():
    """加载并预处理训练数据 - 使用apply_chat_template"""
    ds = load_dataset("json", data_files="cleaned_data/training_data.jsonl", split="train")
    print(f"原始数据: {len(ds)} 条")
    
    # 加载tokenizer用于格式化
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    def format_with_template(example):
        """使用官方chat template格式化对话"""
        messages = []
        
        # 处理你的数据格式
        if "qa_pairs" in example and len(example["qa_pairs"]) > 0:
            conv = example["qa_pairs"][0].get("conversations", [])
            for msg in conv:
                role = msg.get("from", "")
                content = msg.get("value", "")
                
                # 映射角色名称到标准格式
                if role == "system":
                    messages.append({"role": "system", "content": content})
                elif role in ["user", "human"]:
                    messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    messages.append({"role": "assistant", "content": content})
        elif "messages" in example:
            # 如果已经是标准格式
            messages = example["messages"]
        elif "instruction" in example and "output" in example:
            # Alpaca格式转换
            if "input" in example and example["input"]:
                content = f"{example['instruction']}\n\nInput: {example['input']}"
            else:
                content = example["instruction"]
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": example["output"]}
            ]
        
        if not messages:
            return {"text": ""}
        
        # 使用官方template格式化 - 关键修复！
        try:
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            return {"text": text}
        except Exception as e:
            print(f"格式化错误: {e}, messages: {messages}")
            return {"text": ""}
    
    ds = ds.map(format_with_template, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["text"]) > 20)  # 过滤太短的
    
    print(f"有效数据: {len(ds)} 条")
    print("示例数据:")
    print(ds[0]["text"][:500] if len(ds) > 0 else "无数据")
    
    return ds

def load_model():
    """加载模型和tokenizer - 不使用4-bit量化"""
    # 检测支持的数据类型
    supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if supports_bf16 else torch.float16
    
    print(f"使用数据类型: {dtype}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        padding_side="right"
    )
    
    # 确保pad token设置正确
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 关键修复：不使用4-bit量化，改用16-bit
    # Qwen3.5不推荐QLoRA，因为量化误差会导致训练不稳定[^1^]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",  # 自动分配层到GPU/CPU
        # 如果显存不够，可以使用8-bit，但4-bit不推荐
        # load_in_8bit=True,  # 备选方案：8-bit比4-bit稳定
    )
    
    # 准备模型用于训练（如果需要8-bit）
    # if hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
    #     model = prepare_model_for_kbit_training(model)
    
    # 关键修复：扩展target_modules，包含MLP层
    # 对于小样本学习，训练更多层有助于学习领域知识[^3^]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # 增加rank以提升学习能力
        lora_alpha=16,  # 保持alpha=r
        lora_dropout=0.05,  # 小样本用较小dropout
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",  # 添加MLP层
        ],
        bias="none",
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

def train():
    """训练主函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 先加载tokenizer用于数据预处理
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # 准备数据
    train_ds = prepare_dataset()
    
    # 加载模型
    model, tokenizer = load_model()
    
    # 关键修复：保守的训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # 小样本3轮足够
        per_device_train_batch_size=1,  # 小batch稳定
        gradient_accumulation_steps=8,  # 有效batch=8
        learning_rate=1e-4,  # 降低学习率 (从3e-5调整到1e-4，对于LoRA更合理)
        max_grad_norm=1.0,  # 放宽梯度裁剪，避免过度限制
        warmup_ratio=0.1,  # 10% warmup，更平滑启动
        logging_steps=5,  # 每5步记录，避免日志过多
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        fp16=not (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        dataloader_num_workers=0,  # 避免多进程问题
        remove_unused_columns=False,
        optim="adamw_torch",
        weight_decay=0.01,  # 添加正则化
        lr_scheduler_type="cosine",  # 余弦退火
        seed=42,
        report_to="none",
    )
    
    # 关键修复：使用正确的SFTTrainer参数
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        max_seq_length=2048,  # 明确设置最大长度
        # 数据集字段名
        dataset_text_field="text",
        # 不使用packing，避免长度不一致问题
        packing=False,
    )
    
    print("\n" + "="*60)
    print("开始SFT训练 (修复版 v4)...")
    print("="*60)
    print("关键修复:")
    print("  - 移除4-bit量化，使用bf16/fp16 (Qwen3.5不推荐QLoRA)")
    print("  - 使用tokenizer.apply_chat_template处理数据格式")
    print("  - 扩展LoRA target到MLP层 (gate/up/down_proj)")
    print("  - 降低学习率: 3e-5 → 1e-4 (LoRA通常需要更高LR)")
    print("  - 增加warmup: 固定步数 → 10%比例")
    print("  - 放宽梯度裁剪: 0.3 → 1.0")
    print("="*60 + "\n")
    
    # 训练
    train_result = trainer.train()
    
    # 保存结果
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    
    # 保存训练历史
    history = trainer.state.log_history
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n训练完成!")
    print(f"最终loss: {train_result.training_loss:.4f}")

if __name__ == "__main__":
    train()