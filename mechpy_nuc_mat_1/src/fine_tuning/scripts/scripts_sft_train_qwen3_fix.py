#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\
核材料领域 SFT 训练 - Qwen3.5-2B (QLoRA + LoRA 修正版)

超参建议表（两张 RTX A6000 48G，Qwen3.5-2B 基座）

- 典型场景：小样本 SFT（几百 ~ 几千条）+ QLoRA 4bit + LoRA 适配

1. 学习率 lr（LoRA 常用）
   - 推荐范围：5e-5 ~ 2e-4
   - 默认值：1e-4
   - 调整指引：
     * 若 loss 收敛很慢且始终偏高，可适当升至 1.5e-4 ~ 2e-4
     * 若出现震荡 / 发散（loss 上下大幅波动），应降低到 5e-5 左右

2. LoRA 秩 r / alpha
   - 推荐 r：8 ~ 32，alpha ≈ 2 * r
   - 默认：r = 16, alpha = 32
   - 调整指引：
     * r 越大，可训练参数越多，表达能力更强但更容易过拟合且显存占用略升
     * 小数据集（< 1k 样本）建议 r = 8~16；数据较多（> 3k）可尝试 r = 32

3. 梯度累积 steps（gradient_accumulation_steps）
   - A6000 48G 推荐范围：4 ~ 16（batch_size=1 时）
   - 默认：8
   - 调整指引：
     * 显存吃紧时可增大到 12~16，以换取更大的等效 batch
     * 若显存有富余，可降低到 4，提升吞吐

4. 最大序列长度 max_seq_length
   - 推荐范围：1024 ~ 4096
   - 默认：2048
   - 调整指引：
     * 问答对 /对话较短时，可将 max_seq_length 调至 1024 提升速度
     * 若需要保持更多上下文，可升至 3072/4096，同时适当降低
       per_device_train_batch_size 或增加 gradient_accumulation_steps

本脚本默认选取：lr=1e-4, r=16, alpha=32, gradient_accumulation_steps=8,
max_seq_length=2048，适配“两张 RTX A6000 48G” 的小样本核材料 SFT 训练。
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
OUTPUT_DIR = "./output/nuclear_sft_qwen3.5_2b_fix"
DATA_FILE = "cleaned_data/training_data.jsonl"

# 是否使用 tokenizer.apply_chat_template 生成 ChatML（推荐 True）
USE_CHAT_TEMPLATE: bool = True

# 是否启用梯度检查点（节省显存，略微降低吞吐）
USE_GRADIENT_CHECKPOINTING: bool = True

# 最大序列长度
MAX_SEQ_LENGTH: int = 2048


# ========================
# 数据准备
# ========================

def _format_conversation_with_template(
    ex: Dict[str, Any], tokenizer, use_chat_template: bool
) -> Dict[str, str]:
    """将单条样本格式化为单一 text 字段。

    支持两种模式：
    1) USE_CHAT_TEMPLATE=True：使用 tokenizer.apply_chat_template 生成 ChatML
    2) 否则：使用手写 <|im_start|> / <|im_end|> 模板
    """
    if "qa_pairs" not in ex or len(ex["qa_pairs"]) == 0:
        return {"text": ""}

    conv = ex["qa_pairs"][0].get("conversations", []) or []

    if use_chat_template:
        messages: List[Dict[str, str]] = []
        for msg in conv:
            role = msg.get("from", "")
            content = msg.get("value", "") or ""
            if not content:
                continue
            # 映射到 tokenizer 的角色
            if role == "system":
                mapped_role = "system"
            elif role in ("user", "human"):
                mapped_role = "user"
            elif role == "assistant":
                mapped_role = "assistant"
            else:
                # 未知角色统一视作 user，避免丢信息
                mapped_role = "user"
            messages.append({"role": mapped_role, "content": content})

        if not messages:
            return {"text": ""}

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # 回退到手工模板
            use_chat_template = False
        else:
            return {"text": text.strip()}

    # 回退：手写 ChatML 模板
    text_segments: List[str] = []
    for msg in conv:
        role = msg.get("from", "")
        content = msg.get("value", "") or ""
        if not content:
            continue
        if role == "system":
            text_segments.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role in ("user", "human"):
            text_segments.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            text_segments.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        else:
            text_segments.append(f"<|im_start|>user\n{content}<|im_end|>\n")

    return {"text": "".join(text_segments).strip()}


def prepare_dataset(tokenizer) -> Tuple[Any, Any]:
    """加载并预处理训练数据。

    - 从 JSONL 中读取 qa_pairs / conversations
    - 根据 USE_CHAT_TEMPLATE 选择是否走 apply_chat_template
    - 输出字段为 {'text': '...'}，供 SFTTrainer 使用（dataset_text_field="text"）
    """
    ds = load_dataset("json", data_files=DATA_FILE, split="train")
    print(f"原始: {len(ds)} 条")

    def fmt(ex):
        return _format_conversation_with_template(
            ex, tokenizer=tokenizer, use_chat_template=USE_CHAT_TEMPLATE
        )

    ds = ds.map(fmt, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: isinstance(x["text"], str) and len(x["text"]) > 10)
    print(f"有效: {len(ds)} 条")

    # 小样本场景下直接使用全部数据训练，不单独划分验证集
    return ds, ds


# ========================
# 模型与 LoRA / QLoRA 配置
# ========================

def load_model():
    """加载量化基础模型与 LoRA 适配器。

    - QLoRA：NF4 4bit 量化，compute_dtype=bfloat16
    - device_map="auto" 以兼容单卡 / 双卡
    - LoRA: r=16, alpha=32, dropout=0.1
      target_modules: ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]
      （可根据显存与过拟合情况酌情减少模块或降低 r/alpha）
    """
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

    print("加载基础模型（4-bit NF4 量化，单卡模式）...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        device_map="cuda:0",  # 改为单卡模式，避免多卡device_map冲突
    )

    # LoRA 配置 – 建议 r=16/alpha=32，针对小样本保持适度容量
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

    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        # 关闭缓存以适配梯度检查点
        if hasattr(model, "config"):
            model.config.use_cache = False

    model.print_trainable_parameters()
    return model, tok


# ========================
# 训练曲线可视化
# ========================

def plot_training_history(history: List[Dict[str, Any]], save_path: str) -> None:
    """绘制训练历史曲线。

    - 兼容 SFTTrainer 日志字段
    - 对缺失 / 字符串 / 非正数的 loss 做健壮处理
    - 避免对空值或非法值调用 np.exp
    """
    if not history:
        print("没有训练历史数据")
        return

    train_steps: List[int] = []
    train_losses: List[float] = []
    perplexities: List[float] = []
    accuracies: List[float] = []
    lr_steps: List[int] = []
    lrs: List[float] = []

    for log in history:
        if not isinstance(log, dict):
            continue
        step = log.get("step")
        if step is None:
            # 可能是总结性日志，如 {'train_runtime': ...}
            continue

        # 训练 loss（跳过 eval_loss 相关日志）
        if "loss" in log and "eval_loss" not in log:
            try:
                loss_val = float(log["loss"])
            except (TypeError, ValueError):
                loss_val = math.nan

            if loss_val is not None and not math.isnan(loss_val):
                train_steps.append(step)
                train_losses.append(loss_val)
                # 只在 loss > 0 且有限时计算 PPL
                if loss_val > 0:
                    try:
                        ppl = float(np.exp(loss_val))
                    except (OverflowError, FloatingPointError):
                        ppl = float("inf")
                    perplexities.append(ppl)

        # token 级别准确率
        if "mean_token_accuracy" in log:
            try:
                acc = float(log["mean_token_accuracy"])
            except (TypeError, ValueError):
                acc = math.nan
            if not math.isnan(acc):
                accuracies.append(acc)

        # 学习率日志
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss 曲线
    if train_steps and train_losses:
        axes[0, 0].plot(train_steps, train_losses, "b-o", linewidth=2, markersize=4)
        axes[0, 0].set_xlabel("Training Steps", fontsize=12)
        axes[0, 0].set_ylabel("Loss", fontsize=12)
        axes[0, 0].set_title("Loss Curve", fontsize=14, fontweight="bold")
        axes[0, 0].grid(True, alpha=0.3)
        if min(train_losses) > 0:
            axes[0, 0].set_yscale("log")

    # 2. Perplexity 曲线
    if perplexities:
        steps_for_ppl = train_steps[: len(perplexities)]
        axes[0, 1].plot(
            steps_for_ppl, perplexities, "r-o", linewidth=2, markersize=4
        )
        axes[0, 1].set_xlabel("Training Steps", fontsize=12)
        axes[0, 1].set_ylabel("Perplexity", fontsize=12)
        axes[0, 1].set_title("Perplexity Curve", fontsize=14, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)
        if all(p > 0 for p in perplexities):
            axes[0, 1].set_yscale("log")

    # 3. Token Accuracy 曲线
    if train_steps and accuracies:
        steps_for_acc = train_steps[: len(accuracies)]
        axes[1, 0].plot(
            steps_for_acc, accuracies, "g-o", linewidth=2, markersize=4
        )
        axes[1, 0].set_xlabel("Training Steps", fontsize=12)
        axes[1, 0].set_ylabel("Token Accuracy", fontsize=12)
        axes[1, 0].set_title("Accuracy Curve", fontsize=14, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])

    # 4. 学习率曲线
    if lr_steps and lrs:
        axes[1, 1].plot(lr_steps, lrs, "purple", linewidth=2, marker="o", markersize=4)
        axes[1, 1].set_xlabel("Training Steps", fontsize=12)
        axes[1, 1].set_ylabel("Learning Rate", fontsize=12)
        axes[1, 1].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)

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

    # 固定随机种子，便于复现
    set_seed(42)

    model, tok = load_model()
    train_ds, _ = prepare_dataset(tok)

    # 使用 SFTConfig 替代 TrainingArguments
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,  # 降低学习率，避免梯度爆炸
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=20,  # 增加warmup
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,  # A6000 支持 bfloat16
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        dataloader_num_workers=0,
        eval_strategy="no",  # 小样本不单独做 eval
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        max_grad_norm=0.5,  # 更严格的梯度裁剪
        max_length=MAX_SEQ_LENGTH,  # 最大序列长度
        dataset_text_field="text",
    )

    # 定义 formatting_func 替代 dataset_text_field
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
    print("开始 SFT 训练 (Qwen3.5-2B QLoRA + LoRA 修正版)...")
    print("配置要点:")
    print("  - QLoRA: 4bit NF4 + double quant, compute_dtype=bfloat16")
    print("  - LoRA: r=16, alpha=32, dropout=0.1, 目标模块: q/k/v/o/up/down/gate")
    print("  - SFTTrainer: tokenizer=tok, dataset_text_field='text', max_seq_length=2048")
    print("  - data_collator=DataCollatorForLanguageModeling(mlm=False)")
    print("  - device_map='auto' (兼容单/双卡)")
    print("  - lr=1e-4, warmup_ratio=0.05, cosine 调度")
    print("  - per_device_train_batch_size=1, gradient_accumulation_steps=8")
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

    # 保存最终模型（含 LoRA 适配器）
    final_model_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_model_dir)
    print(f"模型已保存到: {final_model_dir}")

    # 提取最后一次训练 loss
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
        "train_runtime": getattr(train_result, "metrics", {}).get("train_runtime", None),
        "train_samples_per_second": getattr(train_result, "metrics", {}).get(
            "train_samples_per_second", None
        ),
    }

    metrics_path = os.path.join(OUTPUT_DIR, "final_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"最终指标已保存到: {metrics_path}")

    if final_loss is not None:
        print("\n==== 最终训练指标 ====")
        print(f"final_train_loss = {final_loss:.4f}")
        if ppl is not None:
            print(f"perplexity       = {ppl:.4f}")
        print("=======================\n")


if __name__ == "__main__":
    # 多卡启动示例（按需选择）：
    # 1) torchrun
    #    torchrun --nproc_per_node=2 scripts/sft_train_qwen3_fix.py
    # 2) accelerate
    #    accelerate launch --multi_gpu --num_processes=2 scripts/sft_train_qwen3_fix.py
    train()
