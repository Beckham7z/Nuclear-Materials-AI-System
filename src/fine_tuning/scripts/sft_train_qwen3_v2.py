#!/usr/bin/env python3
"""核材料领域SFT训练 - Qwen3.5-9B (小样本优化版 v2)"""
import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

MODEL_PATH = "/home/zyx/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B"
OUTPUT_DIR = "./output/nuclear_sft_qwen3.5_9b_v2"

def prepare_dataset():
    """加载并预处理训练数据"""
    ds = load_dataset("json", data_files="cleaned_data/training_data.jsonl", split="train")
    print(f"原始: {len(ds)} 条")
    
    def fmt(ex):
        if "qa_pairs" in ex and len(ex["qa_pairs"]) > 0:
            conv = ex["qa_pairs"][0].get("conversations", [])
            text = ""
            for msg in conv:
                role = msg.get("from", "user")
                content = msg.get("value", "")
                if role == "human":
                    text += f"<|im_start|>user\n{content}<|im_end|>\n"
                else:
                    text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            return {"text": text.strip()}
        return {"text": ""}
    
    ds = ds.map(fmt, remove_columns=ds.column_names).filter(lambda x: len(x["text"]) > 10)
    print(f"有效: {len(ds)} 条")
    
    # 划分训练集和验证集 (8:2)
    ds_split = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = ds_split['train']
    eval_ds = ds_split['test']
    print(f"训练集: {len(train_ds)} 条, 验证集: {len(eval_ds)} 条")
    
    return train_ds, eval_ds

def load_model():
    """加载模型和tokenizer"""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    # 4-bit量化配置
    cfg = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        quantization_config=cfg, 
        device_map="cuda:0"
    )
    
    # LoRA配置 - 小样本优化
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=8,  # 减小rank，从16降到8
        lora_alpha=16,  # 减小alpha，从32降到16
        lora_dropout=0.1,  # 增加dropout防止过拟合
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 减少目标模块
        bias="none", 
        inference_mode=False
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tok

def plot_training_history(history, save_path):
    """绘制训练历史曲线"""
    if not history:
        print("没有训练历史数据")
        return
    
    steps = []
    losses = []
    accuracies = []
    perplexities = []
    eval_losses = []
    eval_accuracies = []
    
    train_steps = []
    train_losses = []
    
    eval_steps = []
    eval_loss_list = []
    eval_acc_list = []
    
    for log in history:
        # 训练日志
        if 'step' in log and 'loss' in log and 'eval_loss' not in log:
            steps.append(log['step'])
            losses.append(log['loss'])
            train_steps.append(log['step'])
            train_losses.append(log['loss'])
            if 'mean_token_accuracy' in log:
                accuracies.append(log['mean_token_accuracy'])
            if 'loss' in log and not np.isnan(log['loss']):
                perplexities.append(np.exp(log['loss']))
        
        # 验证日志
        if 'eval_loss' in log and 'step' in log:
            eval_steps.append(log['step'])
            eval_loss_list.append(log['eval_loss'])
            if 'eval_accuracy' in log:
                eval_acc_list.append(log['eval_accuracy'])
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss曲线 (训练 + 验证)
    if train_steps:
        axes[0, 0].plot(train_steps, train_losses, 'b-o', linewidth=2, markersize=6, label='Train Loss')
    if eval_steps:
        axes[0, 0].plot(eval_steps, eval_loss_list, 'r-s', linewidth=2, markersize=8, label='Eval Loss')
    axes[0, 0].set_xlabel('Training Steps', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Curve (Train vs Eval)', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    if train_losses and min(train_losses) > 0:
        axes[0, 0].set_yscale('log')
    
    # 2. Perplexity曲线
    if perplexities:
        axes[0, 1].plot(train_steps, perplexities, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Training Steps', fontsize=12)
        axes[0, 1].set_ylabel('Perplexity', fontsize=12)
        axes[0, 1].set_title('Perplexity Curve', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
    
    # 3. Token Accuracy曲线 (训练 + 验证)
    if train_steps and accuracies:
        axes[1, 0].plot(train_steps, accuracies, 'g-o', linewidth=2, markersize=6, label='Train Acc')
    if eval_steps and eval_acc_list:
        axes[1, 0].plot(eval_steps, eval_acc_list, 'm-s', linewidth=2, markersize=8, label='Eval Acc')
    axes[1, 0].set_xlabel('Training Steps', fontsize=12)
    axes[1, 0].set_ylabel('Token Accuracy', fontsize=12)
    axes[1, 0].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # 4. 学习率曲线
    lr_steps = []
    lrs = []
    for log in history:
        if 'step' in log and 'learning_rate' in log:
            lr_steps.append(log['step'])
            lrs.append(log['learning_rate'])
    
    if lr_steps:
        axes[1, 1].plot(lr_steps, lrs, 'purple', linewidth=2, marker='o', markersize=6)
        axes[1, 1].set_xlabel('Training Steps', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()

def train():
    """训练主函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_ds, eval_ds = prepare_dataset()
    model, tok = load_model()
    
    # 小样本优化后的训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,  # 增加训练轮数
        per_device_train_batch_size=1,  # 保持批次大小为1避免OOM
        gradient_accumulation_steps=16,  # 保持梯度累积
        learning_rate=5e-5,  # 大幅降低学习率 (从2e-4降到5e-5)
        max_grad_norm=0.5,  # 更严格的梯度裁剪
        logging_steps=1,  # 每步记录日志
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=0,  # 减少数据加载线程
        warmup_steps=10,  # 减少warmup (小数据集不需要太多)
        eval_strategy="epoch",  # 添加验证策略
        load_best_model_at_end=True,  # 加载最佳模型
        metric_for_best_model="eval_loss",  # 基于eval_loss选择最佳模型
        greater_is_better=False,  # loss越低越好
        report_to="none",
        fp16=False,
        gradient_checkpointing=True,  # 启用梯度检查点节省显存
    )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,  # 添加验证集
        processing_class=tok,
    )
    
    print("\n" + "="*50)
    print("开始SFT训练 (小样本优化版 v2)...")
    print("="*50)
    print(f"训练参数优化:")
    print(f"  - 学习率: 2e-4 → 5e-5")
    print(f"  - LoRA rank: 16 → 8")
    print(f"  - LoRA alpha: 32 → 16")
    print(f"  - Dropout: 0.05 → 0.1")
    print(f"  - Warmup: 50 → 10")
    print(f"  - 批次大小: 1 → 2")
    print(f"  - 梯度累积: 16 → 8")
    print(f"  - 添加验证集 (8:2划分)")
    print("="*50 + "\n")
    
    # 开始训练
    train_result = trainer.train()
    
    # 保存训练历史
    history = trainer.state.log_history
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"训练历史已保存到: {history_path}")
    
    # 绘制训练曲线
    plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plot_training_history(history, plot_path)
    
    # 保存模型
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print("\n训练完成!")
    
    # 输出最终评估结果
    final_metrics = trainer.evaluate()
    print("\n最终评估结果:")
    print(f"  - Eval Loss: {final_metrics.get('eval_loss', 'N/A')}")
    if 'eval_loss' in final_metrics and not np.isnan(final_metrics['eval_loss']):
        perplexity = np.exp(final_metrics['eval_loss'])
        print(f"  - Perplexity: {perplexity:.4f}")
    if 'eval_accuracy' in final_metrics:
        print(f"  - Accuracy: {final_metrics.get('eval_accuracy', 'N/A'):.4f}")
    
    # 保存评估结果
    metrics_path = os.path.join(OUTPUT_DIR, "final_metrics.json")
    # 转换numpy类型为Python原生类型
    final_metrics_clean = {}
    for k, v in final_metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            final_metrics_clean[k] = float(v)
        else:
            final_metrics_clean[k] = v
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics_clean, f, indent=2, ensure_ascii=False)
    print(f"评估结果已保存到: {metrics_path}")

if __name__ == "__main__":
    train()
