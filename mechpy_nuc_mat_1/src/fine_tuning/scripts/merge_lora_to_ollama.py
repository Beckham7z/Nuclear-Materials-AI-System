#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将微调后的LoRA适配器与基础模型合并，并推送到Ollama

使用方法:
    python merge_lora_to_ollama.py

前置要求:
    1. 安装了transformers、peft、accelerate
    2. Ollama服务已启动
    3. 基础模型已下载到本地
"""

import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig
import subprocess

# ========================
# 配置参数
# ========================

# 基础模型路径
BASE_MODEL_PATH = "/home/zyx/.cache/modelscope/hub/models/Qwen/Qwen3.5-2B"

# 微调模型路径 (LoRA适配器)
LORA_ADAPTER_PATH = "/home/zyx/A_project/mechpy_nuc_mat/src/fine_tuning/output/nuclear_sft_dapt_swift_fast/v0-20260310-114209/checkpoint-1890"

# 合并后模型保存路径
MERGED_MODEL_PATH = "/home/zyx/A_project/mechpy_nuc_mat/src/fine_tuning/output/nuclear_sft_dapt_swift_fast/v0-20260310-114209/merged_model"

# Ollama模型名称
OLLAMA_MODEL_NAME = "qwen3.5-2b-nuclear-ft"

# Ollama 可执行文件路径
OLLAMA_BIN = "/home/zyx/A_project/mechpy/bin/ollama"


def load_base_model():
    """加载基础模型"""
    print("正在加载基础模型...")
    
    # 4-bit 量化配置（QLoRA）
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("基础模型加载完成！")
    return model, tokenizer


def merge_lora_adapter(base_model, tokenizer):
    """合并LoRA适配器"""
    print("正在合并LoRA适配器...")
    
    # 加载LoRA配置
    peft_config = PeftConfig.from_pretrained(LORA_ADAPTER_PATH)
    
    # 手动设置正确的 target_modules
    # SWIFT框架保存的配置中 target_modules 正则表达式与基础模型结构不匹配
    # 需要手动设置正确的模块名称
    from peft import LoraConfig
    
    # 对于 Qwen3.5-2B，正确的 target_modules 是:
    # - Attention: q_proj, k_proj, v_proj, o_proj
    # - MLP: gate_proj, up_proj, down_proj
    correct_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"]
    
    # 创建新的PeftConfig，使用正确的target_modules
    new_peft_config = LoraConfig(
        task_type=peft_config.task_type,
        r=peft_config.r,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout,
        target_modules=correct_target_modules,
        bias=peft_config.bias,
        modules_to_save=peft_config.modules_to_save,
    )
    
    # 使用新的配置加载PeftModel
    model = PeftModel.from_pretrained(
        base_model, 
        LORA_ADAPTER_PATH,
        config=new_peft_config
    )
    
    # 合并LoRA权重到基础模型
    merged_model = model.merge_and_unload()
    
    print("LoRA适配器合并完成！")
    return merged_model


def save_merged_model(model, tokenizer):
    """保存合并后的模型"""
    print(f"正在保存合并后的模型到: {MERGED_MODEL_PATH}")
    
    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
    
    model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    
    print("模型保存完成！")


def push_to_ollama():
    """将模型推送到Ollama"""
    print(f"正在将模型推送到Ollama，模型名称: {OLLAMA_MODEL_NAME}")
    
    try:
        # 创建Modelfile
        modelfile_content = f"""
FROM {MERGED_MODEL_PATH}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
TEMPLATE \"\"\"{{- range .Messages}}{{ .System}}{{ .Content}} {{ end}}{{ .Input}}\"\"\"
"""
        
        modelfile_path = os.path.join(MERGED_MODEL_PATH, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        
        # 使用指定的ollama路径创建模型
        cmd = [OLLAMA_BIN, "create", OLLAMA_MODEL_NAME, "-f", modelfile_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 模型成功推送到Ollama，名称: {OLLAMA_MODEL_NAME}")
        else:
            print(f"⚠️ ollama create返回: {result.stderr}")
            print("尝试直接推送模型文件...")
            
    except Exception as e:
        print(f"推送过程出错: {e}")
        print("请手动执行以下命令：")
        print(f"1. 先将模型转换为GGUF格式（如果需要）")
        print(f"2. {OLLAMA_BIN} create {OLLAMA_MODEL_NAME} -f <Modelfile路径>")


def check_ollama_model():
    """检查Ollama中是否有该模型"""
    try:
        # 使用指定的ollama路径列出模型
        cmd = [OLLAMA_BIN, "list"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            if OLLAMA_MODEL_NAME in result.stdout:
                print(f"✅ Ollama中已存在模型: {OLLAMA_MODEL_NAME}")
                return True
            else:
                print(f"⚠️ Ollama中未找到模型: {OLLAMA_MODEL_NAME}")
                return False
        else:
            print(f"⚠️ ollama list返回: {result.stderr}")
            return False
    except Exception as e:
        print(f"检查Ollama模型列表失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("LoRA模型合并并推送到Ollama")
    print("=" * 60)
    print(f"基础模型: {BASE_MODEL_PATH}")
    print(f"LoRA适配器: {LORA_ADAPTER_PATH}")
    print(f"合并后模型: {MERGED_MODEL_PATH}")
    print(f"Ollama模型名: {OLLAMA_MODEL_NAME}")
    print("=" * 60)
    
    # 检查是否已存在合并后的模型
    if os.path.exists(os.path.join(MERGED_MODEL_PATH, "config.json")):
        print("检测到已存在的合并模型，跳过合并步骤...")
    else:
        # 1. 加载基础模型
        base_model, tokenizer = load_base_model()
        
        # 2. 合并LoRA适配器
        merged_model = merge_lora_adapter(base_model, tokenizer)
        
        # 3. 保存合并后的模型
        save_merged_model(merged_model, tokenizer)
        
        # 释放内存
        del base_model
        del merged_model
        torch.cuda.empty_cache()
    
    # 4. 推送到Ollama
    if check_ollama_model():
        print("模型已在Ollama中，无需重复推送")
    else:
        push_to_ollama()
    
    print("\n" + "=" * 60)
    print("完成！请在Web界面中使用以下模型名称:")
    print(f"  {OLLAMA_MODEL_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()
