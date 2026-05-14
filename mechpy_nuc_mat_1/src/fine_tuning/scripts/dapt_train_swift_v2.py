#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核材料领域 DAPT 训练 - 使用 Swift 框架 (Python API)
Qwen3.5-4B (继续预训练)

使用方法:
  conda activate nuclear_sft
  python scripts/dapt_train_swift_v2.py
"""

import os
import subprocess

# 设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MAX_PIXELS'] = '1003520'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ========================
# 训练参数配置
# ========================
MODEL = "Qwen/Qwen3.5-2B"  # 使用2B模型，单卡可运行
OUTPUT_DIR = "./output/nuclear_sft_dapt_swift"
DATA_PATH = "dapt_sft_data/dapt_training_data.txt"

# Swift sft 命令参数
# 说明:
#   --tuner_type lora: 使用LoRA微调
#   --dataset: 使用纯文本数据
#   --learning_rate: DAPT使用较低学习率
#   --num_train_epochs: 训练轮数
#   --max_length: 最大序列长度
#   --lora_rank: LoRA rank

args = [
    "swift", "sft",
    "--model", MODEL,
    "--tuner_type", "lora",
    "--dataset", DATA_PATH,
    "--load_from_cache_file", "true",
    "--torch_dtype", "bfloat16",
    "--num_train_epochs", "3",
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--learning_rate", "1e-5",
    "--lora_rank", "8",
    "--lora_alpha", "32",
    "--target_modules", "all-linear",
    "--gradient_accumulation_steps", "8",
    "--output_dir", OUTPUT_DIR,
    "--save_strategy", "epoch",
    "--save_total_limit", "2",
    "--logging_steps", "1",
    "--max_length", "2048",
    "--warmup_steps", "50",
    "--dataset_num_proc", "2",
    "--dataloader_num_workers", "2",
    "--model_author", "Qwen",
    "--model_name", "nuclear-expert",
]


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("开始 DAPT 训练 (Swift框架)")
    print("配置要点:")
    print(f"  - 模型: {MODEL}")
    print(f"  - 数据: {DATA_PATH}")
    print(f"  - 学习率: 1e-5")
    print(f"  - 训练轮数: 3")
    print(f"  - LoRA rank: 8")
    print(f"  - max_length: 2048")
    print("=" * 60)
    print()
    print("执行命令:", " ".join(args))
    print()
    
    # 执行训练
    result = subprocess.run(args)
    
    if result.returncode == 0:
        print("\n训练完成!")
    else:
        print(f"\n训练失败! 返回码: {result.returncode}")


if __name__ == "__main__":
    main()
