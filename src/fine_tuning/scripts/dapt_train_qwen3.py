#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\
核材料领域 DAPT 训练 - Qwen3.5-2B (继续预训练)

DAPT = Domain Adaptive Pre-Training
目的：让通用模型学习核材料领域的专业知识

数据格式：纯文本 (.txt)
训练目标：预测下一个token (causal LM)
"""

import os
import json
import math
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig


# ========================
# 基本路径配置
# ========================
MODEL_PATH = "/home/zyx/.cache/modelscope/hub/models/Qwen/Qwen3.5-2B"
OUTPUT_DIR = "./output/nuclear_sft_dapt"
DATA_FILE = "dapt_sft_data/dapt_training_data.txt"  # DAPT用纯文本

# 最大序列长度
MAX_SEQ_LENGTH: int = 2048


# ========================
# 数据准备
# ========================

def prepare_dataset(tokenizer) -> Tuple[Any, Any]:
    """加载并预处理DAPT训练数据。
    
    DAPT数据是纯文本格式，每行是一段文本
    """
    # 加载纯文本数据
    ds = load_dataset("text", data_files=DATA_FILE, split="train")
    print(f"原始: {len(ds)} 条")
    
    # 过滤过短的文本
    ds = ds.filter(lambda x: len(x["text"]) > 50)
    print(f"有效: {len(ds)} 条")
    
    return ds, ds


# ========================
# 模型配置
# ========================

def load_model():
    """加载量化基础模型与LoRA适配器。"""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 4-bit 量化配置（QLoRA）
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("加载基础模型（4-bit NF4 量化）...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        device_map="cuda:0",
    )

    # LoRA配置 - DAPT可以使用较小的r
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    model.print_trainable_parameters()
    return model, tok


# ========================
# 训练曲线可视化
# ========================

def plot_training_history(history: List[Dict[str, Any]], save_path: str) -> None:
    """绘制训练历史曲线。"""
    if not history:
        print("没有训练历史数据")
        return

    train_steps: List[int] = []
    train_losses: List[float] = []
    perplexities: List[float] = []
    lr_steps: List[int] = []
    lrs: List[float] = []

    for log in history:
        if not isinstance(log, dict):
            continue
        step = log.get("step")
        if step is None:
            continue

        if "loss" in log and "eval_loss" not in log:
            try:
                loss_val = float(log["loss"])
            except (TypeError, ValueError):
                loss_val = math.nan

            if loss_val is not None and not math.isnan(loss_val):
                train_steps.append(step)
                train_losses.append(loss_val)
                if loss_val > 0:
                    try:
                        ppl = float(np.exp(loss_val))
                    except (OverflowError, FloatingPointError):
                        ppl = float("inf")
                    perplexities.append(ppl)

        if "learning_rate" in log:
            try:
                lr_val = float(log["learning_rate"])
            except (TypeError, ValueError):
                lr_val = math.nan
            if not math.isnan(lr_val):
                lr_steps.append(step)
                lrs.append(lr_val)

    if not train_steps:
        print("日志中未找到有效的训练 step / loss 信息，跳过绘图。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss曲线
    if train_steps and train_losses:
        axes[0].plot(train_steps, train_losses, "b-o", linewidth=2, markersize=4)
        axes[0].set_xlabel("Training Steps", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("DAPT Loss Curve", fontsize=14, fontweight="bold")
        axes[0].grid(True, alpha=0.3)

    # Perplexity曲线
    if perplexities:
        steps_for_ppl = train_steps[: len(perplexities)]
        axes[1].plot(steps_for_ppl, perplexities, "r-o", linewidth=2, markersize=4)
        axes[1].set_xlabel("Training Steps", fontsize=12)
        axes[1].set_ylabel("Perplexity", fontsize=12)
        axes[1].set_title("DAPT Perplexity Curve", fontsize=14, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存到: {save_path}")


# ========================
# 训练主流程
# ========================

def train() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 固定随机种子
    set_seed(42)

    model, tok = load_model()
    train_ds, _ = prepare_dataset(tok)

    # SFTConfig配置
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,  # DAPT通常用较低学习率
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        max_grad_norm=0.5,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    # formatting_func
    def formatting_func(examples):
        return examples["text"]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tok,
        formatting_func=formatting_func,
    )

    print("\n" + "=" * 60)
    print("开始 DAPT 训练 (Qwen3.5-2B 继续预训练)...")
    print("配置要点:")
    print("  - 数据: dapt_training_data.txt (纯文本)")
    print("  - 目标: 预测下一个token (领域知识学习)")
    print("  - QLoRA: 4bit NF4 + double quant")
    print("  - LoRA: r=8, alpha=16")
    print("  - max_seq_length: 2048")
    print("=" * 60 + "\n")

    train_result = trainer.train()

    # 训练历史
    history: List[Dict[str, Any]] = trainer.state.log_history
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"训练历史已保存到: {history_path}")

    # 绘制训练曲线
    plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plot_training_history(history, plot_path)

    # 保存最终模型
    final_model_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_model_dir)
    print(f"模型已保存到: {final_model_dir}")

    # 提取最后一次训练loss
    final_loss = None
    for log in reversed(history):
        if not isinstance(log, dict):
            continue
        if "loss" in log and "eval_loss" not in log:
            try:
                final_loss = float(log["loss"])
            except (TypeError, ValueError):
                final_loss = None
            break

    ppl = None
    if final_loss is not None and not math.isnan(final_loss):
        try:
            ppl = float(np.exp(final_loss)) if final_loss > 0 else None
        except (OverflowError, FloatingPointError):
            ppl = None

    metrics = {
        "final_train_loss": float(final_loss) if final_loss is not None else None,
        "perplexity": ppl,
    }

    metrics_path = os.path.join(OUTPUT_DIR, "final_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"最终指标已保存到: {metrics_path}")

    if final_loss is not None:
        print("\n==== DAPT训练完成 ====")
        print(f"final_train_loss = {final_loss:.4f}")
        if ppl is not None:
            print(f"perplexity       = {ppl:.4f}")
        print("=======================\n")


if __name__ == "__main__":
    train()
