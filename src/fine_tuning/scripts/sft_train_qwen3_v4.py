#!/usr/bin/env python3
"""核材料领域SFT训练 - Qwen3.5-2B (小样本优化版 v4)
进一步修复：
1. 添加数据验证和调试
2. 使用更小的学习率
3. 添加gradient norm检查
4. 修复可能的tokenizer问题
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
OUTPUT_DIR = "./output/nuclear_sft_qwen3.5_9b_v4"

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
    
    # 调试：检查数据长度分布
    lengths = [len(x["text"]) for x in ds]
    print(f"文本长度统计: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.0f}")
    
    # 限制最大长度，避免过长导致问题
    max_length = 2048
    ds = ds.filter(lambda x: len(x["text"]) <= max_length)
    print(f"过滤后(长度<={max_length}): {len(ds)} 条")
    
    return ds, ds

def load_model():
    """加载模型和tokenizer"""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None: 
        tok.pad_token = tok.eos_token
        print(f"Set pad_token to eos_token: {tok.pad_token}")
    
    # 打印tokenizer信息用于调试
    print(f"Tokenizer config: pad={tok.pad_token}, eos={tok.eos_token}, bos={tok.bos_token}")
    print(f"Vocab size: {len(tok)}")
    
    # 4-bit量化配置 - 使用更安全的设置
    cfg = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_compute_dtype=torch.float16  # 使用fp16而不是bf16，更稳定
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        quantization_config=cfg, 
        device_map="cuda:0",
        torch_dtype=torch.float16
    )
    
    # 锁定模型参数，只训练LoRA
    for param in model.parameters():
        param.requires_grad = False
    
    # LoRA配置 - 进一步降低复杂度
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=4,  # 进一步减小rank
        lora_alpha=8,  # 减小alpha
        lora_dropout=0.15,  # 增加dropout
        target_modules=["q_proj", "k_proj", "v_proj"],  # 减少目标模块
        bias="none", 
        inference_mode=False
    )
    model = get_peft_model(model, lora)
    
    # 确保只有LoRA参数可训练
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    
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
            # 跳过验证loss和无效loss
            if 'eval_loss' not in log and log['loss'] is not None and not np.isnan(log['loss']) and log['loss'] > 0:
                train_steps.append(log['step'])
                train_losses.append(log['loss'])
                if 'mean_token_accuracy' in log and log['mean_token_accuracy'] is not None:
                    accuracies.append(log['mean_token_accuracy'])
                perplexities.append(np.exp(log['loss']))
        # 学习率
        if 'step' in log and 'learning_rate' in log and log['learning_rate'] is not None:
            lr_steps.append(log['step'])
            lrs.append(log['learning_rate'])
    
    if not train_steps:
        print("没有有效的训练数据")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss曲线
    axes[0, 0].plot(train_steps, train_losses, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Training Steps', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Perplexity曲线
    axes[0, 1].plot(train_steps, perplexities, 'r-o', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Training Steps', fontsize=12)
    axes[0, 1].set_ylabel('Perplexity', fontsize=12)
    axes[0, 1].set_title('Perplexity Curve', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Token Accuracy曲线
    if accuracies:
        axes[1, 0].plot(train_steps[:len(accuracies)], accuracies, 'g-o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Training Steps', fontsize=12)
    axes[1, 0].set_ylabel('Token Accuracy', fontsize=12)
    axes[1, 0].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # 4. 学习率曲线
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
    
    # 优化后的训练参数 - 进一步优化稳定性
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,  # 进一步减少
        learning_rate=1e-5,  # 大幅降低学习率
        max_grad_norm=0.1,  # 更严格的梯度裁剪
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,  # 使用fp16而不是bf16
        bf16=False,
        dataloader_num_workers=0,
        warmup_steps=3,  # 减少warmup
        eval_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        optim="adamw_torch",
        logging_dir=f"{OUTPUT_DIR}/logs",
        # 添加更多稳定训练选项
        dataloader_pin_memory=False,
        # 跳过可能导致问题的步骤
        max_steps=100,  # 限制最大步数
    )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tok,
        # 禁用packing，避免潜在问题
        packing=False,
    )
    
    print("\n" + "="*50)
    print("开始SFT训练 (小样本优化版 v4)...")
    print("="*50)
    print(f"修复内容:")
    print(f"  - 正确处理system/user/assistant角色")
    print(f"  - 移除验证集，使用全部数据训练")
    print(f"  - 梯度累积: 4 → 2")
    print(f"  - 学习率: 3e-5 → 1e-5")
    print(f"  - 梯度裁剪: 0.3 → 0.1")
    print(f"  - LoRA rank: 8 → 4")
    print(f"  - 使用fp16替代bf16")
    print(f"  - 添加最大长度限制(2048)")
    print(f"  - 禁用packing")
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
            if 'loss' in log and 'eval_loss' not in log and log.get('loss', 0) > 0:
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
