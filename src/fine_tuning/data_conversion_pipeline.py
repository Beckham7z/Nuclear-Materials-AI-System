"""
数据转换管道：将LightRAG输出的JSON文件转换为DAPT和SFT训练格式

DAPT格式：纯文本格式，每行一个文档，用于继续预训练
SFT格式：Alpaca/ShareGPT格式，用于有监督微调
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import glob


class DataConversionPipeline:
    def __init__(self, input_dir: str = "cleaned_data", output_dir: str = "dapt_sft_data"):
        """
        初始化数据转换管道
        
        Args:
            input_dir: 输入JSON文件目录
            output_dir: 输出目录
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载所有JSON文件
        self.json_files = list(self.input_dir.glob("*.json"))
        print(f"找到 {len(self.json_files)} 个JSON文件")
        
    def load_json_data(self, filepath: Path) -> List[Dict]:
        """加载单个JSON文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  - 加载 {filepath.name}: {len(data)} 条记录")
        return data
    
    def extract_text_from_qa(self, qa_data: Dict) -> List[str]:
        """
        从QA数据中提取文本
        
        Returns:
            文本列表
        """
        texts = []
        
        # 提取所有对话内容
        if 'qa_pairs' in qa_data:
            for qa_pair in qa_data['qa_pairs']:
                if 'conversations' in qa_pair:
                    for conv in qa_pair['conversations']:
                        if 'value' in conv and conv['value'].strip():
                            texts.append(conv['value'].strip())
        
        return texts
    
    def convert_to_dapt_format(self, data: List[Dict]) -> List[str]:
        """
        转换为DAPT格式（纯文本）
        
        DAPT格式：每个文档是一个纯文本块，用于继续预训练
        模型学习预测下一个token，学习领域知识
        """
        dapt_data = []
        
        for item in data:
            # 收集所有相关文本
            texts = self.extract_text_from_qa(item)
            
            if texts:
                # 将所有文本合并为一个文档
                # 可以选择不同的合并策略
                combined_text = "\n\n".join(texts)
                
                # 过滤太短的文本
                if len(combined_text) > 50:
                    dapt_data.append(combined_text)
        
        print(f"  生成 DAPT 数据: {len(dapt_data)} 条")
        return dapt_data
    
    def convert_to_sft_format_alpaca(self, data: List[Dict]) -> List[Dict]:
        """
        转换为Alpaca格式
        
        Alpaca格式:
        {
            "instruction": "指令",
            "input": "输入（可为空）",
            "output": "输出"
        }
        """
        sft_data = []
        
        for item in data:
            if 'qa_pairs' not in item:
                continue
                
            for qa_pair in item['qa_pairs']:
                if 'conversations' not in qa_pair:
                    continue
                
                conversations = qa_pair['conversations']
                
                # 提取user和assistant对话
                user_msg = None
                assistant_msg = None
                system_msg = None
                
                for conv in conversations:
                    role = conv.get('from', '')
                    content = conv.get('value', '').strip()
                    
                    if role == 'system':
                        system_msg = content
                    elif role == 'user' and not user_msg:
                        user_msg = content
                    elif role == 'assistant' and not assistant_msg:
                        assistant_msg = content
                
                # 只有当有user和assistant时才添加
                if user_msg and assistant_msg:
                    # 构建instruction
                    if system_msg:
                        instruction = f"{system_msg}\n\n{user_msg}"
                    else:
                        instruction = user_msg
                    
                    sft_data.append({
                        "instruction": instruction,
                        "input": "",
                        "output": assistant_msg
                    })
        
        print(f"  生成 Alpaca SFT 数据: {len(sft_data)} 条")
        return sft_data
    
    def convert_to_sft_format_sharegpt(self, data: List[Dict]) -> List[Dict]:
        """
        转换为ShareGPT格式
        
        ShareGPT格式:
        {
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."}
            ]
        }
        """
        sft_data = []
        
        for item in data:
            if 'qa_pairs' not in item:
                continue
                
            for qa_pair in item['qa_pairs']:
                if 'conversations' not in qa_pair:
                    continue
                
                conversations = qa_pair['conversations']
                
                # 转换格式
                converted_conv = []
                for conv in conversations:
                    role = conv.get('from', '')
                    content = conv.get('value', '').strip()
                    
                    if not content:
                        continue
                    
                    # 映射角色
                    if role == 'system':
                        converted_conv.append({"from": "system", "value": content})
                    elif role == 'user':
                        converted_conv.append({"from": "human", "value": content})
                    elif role == 'assistant':
                        converted_conv.append({"from": "gpt", "value": content})
                
                # 只有当有有效对话时才添加
                if len(converted_conv) >= 2:
                    sft_data.append({
                        "conversations": converted_conv
                    })
        
        print(f"  生成 ShareGPT SFT 数据: {len(sft_data)} 条")
        return sft_data
    
    def process_all_files(self):
        """处理所有JSON文件并生成各种格式"""
        
        # 合并所有数据
        all_data = []
        for json_file in self.json_files:
            data = self.load_json_data(json_file)
            all_data.extend(data)
        
        print(f"\n总计加载 {len(all_data)} 条记录")
        
        # 1. 生成DAPT格式
        print("\n=== 生成 DAPT 格式 ===")
        dapt_texts = self.convert_to_dapt_format(all_data)
        
        # 保存DAPT格式（每行一个文档）
        dapt_output = self.output_dir / "dapt_training_data.txt"
        with open(dapt_output, 'w', encoding='utf-8') as f:
            for text in dapt_texts:
                f.write(text + "\n\n")
        print(f"  保存至: {dapt_output}")
        
        # 2. 生成Alpaca格式
        print("\n=== 生成 Alpaca SFT 格式 ===")
        alpaca_data = self.convert_to_sft_format_alpaca(all_data)
        
        alpaca_output = self.output_dir / "sft_training_alpaca.json"
        with open(alpaca_output, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
        print(f"  保存至: {alpaca_output}")
        
        # 3. 生成ShareGPT格式
        print("\n=== 生成 ShareGPT SFT 格式 ===")
        sharegpt_data = self.convert_to_sft_format_sharegpt(all_data)
        
        sharegpt_output = self.output_dir / "sft_training_sharegpt.json"
        with open(sharegpt_output, 'w', encoding='utf-8') as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
        print(f"  保存至: {sharegpt_output}")
        
        # 打印统计信息
        print("\n" + "="*50)
        print("数据转换完成！")
        print("="*50)
        print(f"DAPT 纯文本数据: {len(dapt_texts)} 条")
        print(f"Alpaca SFT 数据: {len(alpaca_data)} 条")
        print(f"ShareGPT SFT 数据: {len(sharegpt_data)} 条")
        
        return {
            "dapt_count": len(dapt_texts),
            "alpaca_count": len(alpaca_data),
            "sharegpt_count": len(sharegpt_data)
        }
    
    def generate_train_val_split(self, sft_format: str = "alpaca", train_ratio: float = 0.9):
        """
        生成训练集和验证集划分
        
        Args:
            sft_format: 'alpaca' 或 'sharegpt'
            train_ratio: 训练集比例
        """
        if sft_format == "alpaca":
            input_file = self.output_dir / "sft_training_alpaca.json"
        else:
            input_file = self.output_dir / "sft_training_sharegpt.json"
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 随机打乱
        import random
        random.shuffle(data)
        
        # 划分
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # 保存
        if sft_format == "alpaca":
            train_file = self.output_dir / "sft_train_alpaca.json"
            val_file = self.output_dir / "sft_val_alpaca.json"
        else:
            train_file = self.output_dir / "sft_train_sharegpt.json"
            val_file = self.output_dir / "sft_val_sharegpt.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n训练集: {len(train_data)} 条 -> {train_file}")
        print(f"验证集: {len(val_data)} 条 -> {val_file}")
        
        return train_data, val_data


def main():
    """主函数"""
    print("="*60)
    print("核材料领域数据转换管道")
    print("="*60)
    
    # 创建转换管道
    pipeline = DataConversionPipeline(
        input_dir="cleaned_data",
        output_dir="dapt_sft_data"
    )
    
    # 处理所有文件
    stats = pipeline.process_all_files()
    
    # 生成训练/验证集划分
    print("\n=== 生成训练/验证集划分 ===")
    pipeline.generate_train_val_split(sft_format="alpaca", train_ratio=0.9)
    pipeline.generate_train_val_split(sft_format="sharegpt", train_ratio=0.9)
    
    print("\n" + "="*60)
    print("所有转换任务完成！")
    print("="*60)


if __name__ == "__main__":
    main()
