#!/usr/bin/env python3
"""
从 myKG/kv_store_full_docs.json 生成 mech_text_chunks_no_dup.csv 文件
该脚本提取文档内容，进行去重处理，并生成适合问答对生成的CSV文件
"""

import json
import pandas as pd
import hashlib
from typing import List, Dict, Set
import re
import os

def load_json_file(file_path: str) -> dict:
    print(f"正在加载文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)          # 一次性加载整个 JSON 对象
    print(f"成功加载 {len(data)} 个文档")
    return data

def extract_text_content(content: str) -> str:
    """从文档内容中提取实际文本"""
    # 移除PDF文件头信息
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # 跳过PDF文件信息行
        if line.startswith('PDF文件:') or line.startswith('文件大小:') or '内容待处理' in line:
            continue
        # 跳过空行和纯空格行
        if line.strip():
            cleaned_lines.append(line.strip())
    
    return ' '.join(cleaned_lines)

def calculate_content_hash(text: str) -> str:
    """计算文本内容的哈希值用于去重"""
    # 标准化文本：小写、移除多余空格
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def process_documents(docs_data: Dict) -> List[Dict]:
    """处理文档数据，提取内容并去重"""
    processed_chunks = []
    seen_hashes: Set[str] = set()
    
    for doc_id, doc_info in docs_data.items():
        if not isinstance(doc_info, dict):
            continue
            
        content = doc_info.get('content', '')
        if not content:
            continue
            
        # 提取实际文本内容
        text_content = extract_text_content(content)
        if not text_content or len(text_content) < 50:  # 太短的内容跳过
            continue
        
        # 计算哈希值去重
        content_hash = calculate_content_hash(text_content)
        if content_hash in seen_hashes:
            continue
        
        seen_hashes.add(content_hash)
        
        # 创建文本块记录
        chunk = {
            '_id': doc_id,
            'content': text_content,
            'file_path': doc_info.get('file_path', 'unknown'),
            'full_doc_id': doc_id,
            'tokens': len(text_content.split()),  # 简单分词估算
            'chunk_order_index': 0,  # 默认值
            'content_hash': content_hash
        }
        
        processed_chunks.append(chunk)
    
    print(f"处理完成，共 {len(processed_chunks)} 个唯一文本块")
    return processed_chunks

def save_to_csv(chunks: List[Dict], output_path: str):
    """保存处理后的文本块到CSV文件"""
    if not chunks:
        print("没有数据可保存")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(chunks)
    
    # 确保必要的列存在
    required_columns = ['_id', 'content', 'file_path', 'full_doc_id', 'tokens', 'chunk_order_index']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    
    # 重新排列列顺序
    df = df[required_columns]
    
    # 保存到CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"已保存到: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"记录数: {len(df)}")

def main():
    """主函数"""
    # 输入文件路径
    input_file = "/home/zyx/A_project/mechpy_nuc_mat/myKG/kv_store_full_docs.json"
    # 输出文件路径
    output_file = "mech_text_chunks_no_dup.csv"
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        print("请确保文件路径正确")
        return
    
    # 1. 加载JSON数据
    docs_data = load_json_file(input_file)
    if not docs_data:
        print("无法加载数据，请检查文件格式")
        return
    
    # 2. 处理文档并去重
    print("正在处理文档并去重...")
    chunks = process_documents(docs_data)
    
    if not chunks:
        print("没有可用的文本内容")
        return
    
    # 3. 保存到CSV
    print(f"正在保存到 {output_file}...")
    save_to_csv(chunks, output_file)
    
    # 4. 显示统计信息
    print("\n=== 统计信息 ===")
    print(f"总文档数: {len(docs_data)}")
    print(f"唯一文本块数: {len(chunks)}")
    print(f"去重率: {(1 - len(chunks)/len(docs_data))*100:.1f}%" if docs_data else "N/A")
    
    # 显示前几个文本块的示例
    if chunks:
        print("\n=== 前3个文本块示例 ===")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n示例 {i+1}:")
            print(f"ID: {chunk['_id']}")
            print(f"内容预览: {chunk['content'][:100]}...")
            print(f"词数: {chunk['tokens']}")

def create_sample_csv():
    """创建示例CSV文件（如果主文件处理失败）"""
    sample_data = [
        {
            '_id': 'doc-001',
            'content': '核材料在反应堆中受到中子辐照会产生缺陷，这些缺陷会影响材料的力学性能和尺寸稳定性。研究表明，辐照损伤会导致材料硬化和脆化。',
            'file_path': 'nuclear_material_1.pdf',
            'full_doc_id': 'doc-001',
            'tokens': 45,
            'chunk_order_index': 0
        },
        {
            '_id': 'doc-002',
            'content': '核燃料包壳材料需要具有良好的抗腐蚀性能和高温强度。锆合金因其低中子吸收截面和良好的机械性能而被广泛使用。',
            'file_path': 'nuclear_material_2.pdf',
            'full_doc_id': 'doc-002',
            'tokens': 38,
            'chunk_order_index': 0
        },
        {
            '_id': 'doc-003',
            'content': '核废料处理是核能利用的关键环节。玻璃固化技术可以将高放废料固定在玻璃基质中，防止放射性核素迁移。',
            'file_path': 'nuclear_waste_1.pdf',
            'full_doc_id': 'doc-003',
            'tokens': 35,
            'chunk_order_index': 0
        }
    ]
    
    df = pd.DataFrame(sample_data)
    output_file = "mech_text_chunks_no_dup_sample.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"已创建示例文件: {output_file}")
    return output_file

if __name__ == "__main__":
    print("=== 开始生成 mech_text_chunks_no_dup.csv ===")
    try:
        main()
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        print("正在创建示例文件...")
        create_sample_csv()
    print("=== 处理完成 ===")
