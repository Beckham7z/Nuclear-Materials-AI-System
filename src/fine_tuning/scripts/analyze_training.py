#!/usr/bin/env python3
"""
训练结果分析脚本
用于量化和可视化SFT训练结果
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_training_history(history_path):
    """加载训练历史"""
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metrics(history):
    """提取训练指标"""
    train_metrics = {'steps': [], 'loss': [], 'accuracy': [], 'lr': [], 'perplexity': []}
    eval_metrics = {'steps': [], 'loss': [], 'accuracy': [], 'perplexity': []}
    
    for log in history:
        step = log.get('step', 0)
        
        # 训练指标
        if 'loss' in log and 'step' in log:
            train_metrics['steps'].append(step)
            train_metrics['loss'].append(log['loss'])
            if log['loss'] > 0 and not np.isnan(log['loss']):
                train_metrics['perplexity'].append(np.exp(log['loss']))
            else:
                train_metrics['perplexity'].append(float('inf'))
            
            if 'mean_token_accuracy' in log:
                train_metrics['accuracy'].append(log['mean_token_accuracy'])
            if 'learning_rate' in log:
                train_metrics['lr'].append(log['learning_rate'])
        
        # 验证指标
        if 'eval_loss' in log:
            eval_metrics['steps'].append(step)
            eval_metrics['loss'].append(log['eval_loss'])
            if 'eval_loss' in log and log['eval_loss'] > 0:
                eval_metrics['perplexity'].append(np.exp(log['eval_loss']))
            else:
                eval_metrics['perplexity'].append(float('inf'))
            if 'eval_accuracy' in log:
                eval_metrics['accuracy'].append(log['eval_accuracy'])
    
    return train_metrics, eval_metrics


def plot_comprehensive_curves(train_metrics, eval_metrics, save_path):
    """绘制综合训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SFT Training Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Training Loss
    if train_metrics['steps']:
        axes[0, 0].plot(train_metrics['steps'], train_metrics['loss'], 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
