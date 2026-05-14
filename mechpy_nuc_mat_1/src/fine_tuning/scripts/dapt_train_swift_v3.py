#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核材料领域 DAPT 训练 - 加速优化版
"""

import os
import subprocess

# 设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ========================
# 训练参数配置
# ========================
MODEL = "Qwen/Qwen3.5-2B"
OUTPUT_DIR = "./output/nuclear_sft_dapt_swift_fast"
DATA_PATH = "dapt_sft_data/dapt_training_data.txt"

# ========================
# 加速训练配置
# ========================
args = [
    "swift", "sft",
    "--model", MODEL,
    "--tuner_type", "lora",
    "--dataset", DATA_PATH,
    "--load_from_cache_file", "true",
    "--torch_dtype", "bfloat16",
    
    # 1. 减少训练轮数 (3 → 1)
    "--num_train_epochs", "1",
    
    # 2. 增加batch size（如果显存允许）
    "--per_device_train_batch_size", "2",      # 从1提到2
    "--gradient_accumulation_steps", "4",       # 从8降到4
    
    # 3. 提高学习率加速收敛
    "--learning_rate", "2e-5",                   # 从1e-5提到2e-5
    
    # 4. 减少warmup steps
    "--warmup_steps", "20",                      # 从50降到20
    
    # 5. LoRA配置 (可以稍微提高rank保持效果)
    "--lora_rank", "16",                         # 从8提到16
    "--lora_alpha", "32",
    "--target_modules", "all-linear",
    
    # 6. 输出配置
    "--output_dir", OUTPUT_DIR,
    "--save_strategy", "epoch",
    "--save_total_limit", "1",
    "--logging_steps", "10",                      # 减少日志频率
    
    "--max_length", "1024",                       # 降低序列长度（如果数据允许）
    "--dataset_num_proc", "4",
    "--dataloader_num_workers", "4",
    
    # 7. 添加加速选项
    "--gradient_checkpointing", "true",            # 节省显存
    "--optim", "adamw_torch_fused",                # 使用fused优化器加速
    "--model_author", "Qwen",
    "--model_name", "nuclear-expert-fast",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("DAPT 训练 (加速优化版)")
    print("配置要点:")
    print(f"  - 模型: {MODEL}")
    print(f"  - 数据: {DATA_PATH}")
    print(f"  - 训练轮数: 1 (原3轮)")
    print(f"  - 学习率: 2e-5 (原1e-5)")
    print(f"  - Batch size: 2 * 4 = 8 (原 1*8=8)")
    print(f"  - LoRA rank: 16 (原8)")
    print(f"  - max_length: 1024 (原2048)")
    print("=" * 60)
    print()
    print("执行命令:", " ".join(args))
    print()
    
    result = subprocess.run(args)
    
    if result.returncode == 0:
        print("\n✅ 训练完成!")
    else:
        print(f"\n❌ 训练失败! 返回码: {result.returncode}")


if __name__ == "__main__":
    main()