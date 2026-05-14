#!/usr/bin/env python3
"""核材料领域SFT训练 - Qwen3.5-2B (小样本优化版 v3)
修复内容:
1. 修复数据格式处理，正确识别system/user/assistant角色
2. 小数据集不使用验证集，使用全部数据训练
3. 减少梯度累积步数，降低显存占用
4. 优化训练参数
"""
import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

MODEL_PATH = "/home/zyx/.cache/modelscope/hub/models/Qwen/Qwen3.5-2B"
OUTPUT_DIR = "./output/nuclear_sft_qwen3.5_9b_v3"

def prepare_dataset():
    """加载并预处理训练数据"""
    ds = load_dataset("json", data_files="cleaned_data/training_data.jsonl", split="train")
    print(f"原始: {len(ds)} 条")
    
    def fmt(ex):
        if "qa_pairs" in ex and len(ex["qa_pairs"]) > 0:
            conv = ex["qa_pairs"][0].get("conversations", [])
            text = ""
            for msg in conv:
                role = msg.get("from", "")
                content = msg.get("value", "")
                # 正确处理所有角色
                if role == "system":
                    text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user" or role == "human":
                    text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            return {"text": text.strip()}
        return {"text": ""}
    
    ds = ds.map(fmt, remove_columns=ds.column_names).filter(lambda x: len(x["text"]) > 10)
    print(f"有效: {len(ds)} 条")
    
    # 小样本数据不使用验证集，直接返回全部数据用于训练
    # 这样可以最大化利用有限的数据
    return ds, ds

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
        r=16,  # 减小rank
        lora_alpha=32,  # 减小alpha
        lora_dropout=0.05,  # 增加dropout防止过拟合
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
    
    train_steps = []
    train_losses = []
    perplexities = []
    accuracies = []
    lr_steps = []
    lrs = []
    
    for log in history:
        if 'step' in log and 'loss' in log:
            # 跳过验证loss
            if 'eval_loss' not in log:
                train_steps.append(log['step'])
                train_losses.append(log['loss'])
                if 'mean_token_accuracy' in log:
                    accuracies.append(log['mean_token_accuracy'])
                if 'loss' in log and not np.isnan(log['loss']) and log['loss'] > 0:
                    perplexities.append(np.exp(log['loss']))
        # 学习率
        if 'step' in log and 'learning_rate' in log:
            lr_steps.append(log['step'])
            lrs.append(log['learning_rate'])
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss曲线
    if train_steps and train_losses:
        axes[0, 0].plot(train_steps, train_losses, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Training Steps', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        if min(train_losses) > 0:
            axes[0, 0].set_yscale('log')
    
    # 2. Perplexity曲线
    if perplexities:
        axes[0, 1].plot(train_steps[:len(perplexities)], perplexities, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Training Steps', fontsize=12)
        axes[0, 1].set_ylabel('Perplexity', fontsize=12)
        axes[0, 1].set_title('Perplexity Curve', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
    
    # 3. Token Accuracy曲线
    if train_steps and accuracies:
        axes[1, 0].plot(train_steps[:len(accuracies)], accuracies, 'g-o', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Training Steps', fontsize=12)
        axes[1, 0].set_ylabel('Token Accuracy', fontsize=12)
        axes[1, 0].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
    
    # 4. 学习率曲线
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
    
    # 优化后的训练参数 - 针对小样本数据
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,  # 小数据集训练3轮足够
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # 减少梯度累积，降低显存
        learning_rate=2e-5,  # 较低的学习率
        max_grad_norm=0.3,  # 更严格的梯度裁剪防止梯度爆炸
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=0,
        warmup_steps=5,  # 减少warmup
        # 不使用验证集评估，避免OOM
        eval_strategy="no",
        report_to="none",
        fp16=False,
        gradient_checkpointing=True,  # 启用梯度检查点
        # 优化显存
        remove_unused_columns=False,
        optim="adamw_torch",
        # 减少日志内存占用
        log_level="warning",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        # 不传入eval_dataset避免验证时OOM
        processing_class=tok,
    )
    
    print("\n" + "="*50)
    print("开始SFT训练 (小样本优化版 v3)...")
    print("="*50)
    print(f"修复内容:")
    print(f"  - 正确处理system/user/assistant角色")
    print(f"  - 移除验证集，使用全部数据训练")
    print(f"  - 梯度累积: 16 → 4")
    print(f"  - 学习率: 5e-5 → 3e-5")
    print(f"  - 训练轮数: 5 → 3")
    print(f"  - 梯度裁剪: 0.5 → 0.3")
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
    
    # 输出训练损失作为最终指标
    final_loss = None
    if history:
        for log in reversed(history):
            if 'loss' in log and 'eval_loss' not in log and 'loss' in log:
                final_loss = log['loss']
                break
    
    if final_loss:
        print(f"\n最终训练 Loss: {final_loss:.4f}")
        print(f"Perplexity: {np.exp(final_loss):.4f}")
    
    # 保存最终指标
    metrics_path = os.path.join(OUTPUT_DIR, "final_metrics.json")
    final_metrics = {
        "final_train_loss": float(final_loss) if final_loss else None,
        "perplexity": float(np.exp(final_loss)) if final_loss else None,
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f"评估结果已保存到: {metrics_path}")

if __name__ == "__main__":
    train()
