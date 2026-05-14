#!/usr/bin/env python3
"""核材料领域SFT训练 - Qwen3.5-9B"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

MODEL_PATH = "/home/zyx/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B"
OUTPUT_DIR = "./output/nuclear_sft_qwen3.5_9b"

def prepare_dataset():
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
    return ds

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, 
                                                   quantization_config=cfg, device_map="cuda:0")
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                      bias="none", inference_mode=False)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tok

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds = prepare_dataset()
    model, tok = load_model()
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=3,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=2e-4, max_grad_norm=1.0, logging_steps=5,
        save_strategy="epoch", save_total_limit=2, bf16=True,
        dataloader_num_workers=4, warmup_steps=50, report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
    )
    
    print("\n" + "="*50 + "\n开始SFT训练...\n" + "="*50)
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print("\n训练完成!")

if __name__ == "__main__": train()
