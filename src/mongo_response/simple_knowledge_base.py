import os
import json
import time
from typing import List, Dict, Any
import re
import math

def create_knowledge_base_from_markdown(md_file_path: str, output_dir: str):
    """从Markdown文件创建知识库"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取Markdown文件
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📖 读取Markdown文件: {md_file_path}")
    print(f"📊 文件大小: {len(content)} 字符")
    
    # 分割文档为块
    chunks = chunk_markdown_content(content)
    print(f"📄 分割为 {len(chunks)} 个文档块")
    
    # 保存知识库
    knowledge_base = {
        "source_file": md_file_path,
        "created_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        "total_chunks": len(chunks),
        "chunks": chunks
    }
    
    # 保存为JSON文件
    kb_file = os.path.join(output_dir, "knowledge_base.json")
    with open(kb_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    
    print(f"💾 知识库已保存: {kb_file}")
    
    # 创建索引文件
    create_search_index(chunks, output_dir)
    
    return knowledge_base

def chunk_markdown_content(content: str, max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """将Markdown内容分割为块"""
    
    # 按标题分割
    blocks = re.split(r'(?=^#{1,6} )', content, flags=re.MULTILINE)
    blocks = [block.strip() for block in blocks if block.strip()]
    
    chunks = []
    chunk_id = 0
    
    for block in blocks:
        # 如果块太大，进一步分割
        if len(block) > max_chunk_size:
            # 按段落分割
            paragraphs = re.split(r'\n\n+', block)
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append({
                            "id": chunk_id,
                            "content": current_chunk.strip(),
                            "length": len(current_chunk),
                            "type": "paragraph_chunk"
                        })
                        chunk_id += 1
                    current_chunk = para + "\n\n"
            
            if current_chunk.strip():
                chunks.append({
                    "id": chunk_id,
                    "content": current_chunk.strip(),
                    "length": len(current_chunk),
                    "type": "paragraph_chunk"
                })
                chunk_id += 1
        else:
            # 直接使用整个块
            chunks.append({
                "id": chunk_id,
                "content": block,
                "length": len(block),
                "type": "header_block"
            })
            chunk_id += 1
    
    return chunks

def create_search_index(chunks: List[Dict[str, Any]], output_dir: str):
    """创建搜索索引"""
    
    # 创建关键词索引
    keyword_index = {}
    
    for chunk in chunks:
        content = chunk['content'].lower()
        
        # 提取关键词（简单的基于频率的方法）
        words = re.findall(r'\b[a-z]{3,}\b', content)
        word_freq = {}
        
        for word in words:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
        
        # 只保留频率较高的词
        for word, freq in word_freq.items():
            if freq >= 2:  # 至少出现2次
                if word not in keyword_index:
                    keyword_index[word] = []
                keyword_index[word].append(chunk['id'])
    
    # 保存索引
    index_file = os.path.join(output_dir, "search_index.json")
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(keyword_index, f, indent=2, ensure_ascii=False)
    
    print(f"🔍 搜索索引已创建: {index_file}")
    print(f"📝 索引包含 {len(keyword_index)} 个关键词")

def search_knowledge_base(query: str, knowledge_base_path: str):
    """在知识库中搜索"""
    
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)
    
    # 简单的关键词匹配搜索
    query_words = re.findall(r'\b[a-z]{3,}\b', query.lower())
    relevant_chunks = []
    
    for chunk in kb['chunks']:
        content_lower = chunk['content'].lower()
        match_score = sum(1 for word in query_words if word in content_lower)
        
        if match_score > 0:
            relevant_chunks.append({
                'chunk': chunk,
                'score': match_score
            })
    
    # 按匹配分数排序
    relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
    
    return relevant_chunks[:5]  # 返回前5个最相关的结果

def main():
    """主函数"""
    
    md_file_path = "/home/beckham7/A_project/n_material_file/converted_output/converted.md"
    output_dir = "/home/beckham7/A_project/n_material_file/knowledge_base"
    
    print("=" * 60)
    print("🧠 知识库构建器")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(md_file_path):
        print(f"❌ 文件不存在: {md_file_path}")
        return
    
    try:
        # 构建知识库
        kb = create_knowledge_base_from_markdown(md_file_path, output_dir)
        
        print("\n" + "=" * 60)
        print("✅ 知识库构建完成!")
        print("=" * 60)
        print(f"📊 统计信息:")
        print(f"   - 源文件: {kb['source_file']}")
        print(f"   - 创建时间: {kb['created_time']}")
        print(f"   - 文档块数量: {kb['total_chunks']}")
        print(f"   - 输出目录: {output_dir}")
        
        # 测试搜索功能
        print(f"\n🔍 测试搜索功能:")
        test_queries = [
            "nanocrystalline grains",
            "Xe irradiation",
            "hardness measurement"
        ]
        
        for query in test_queries:
            results = search_knowledge_base(query, os.path.join(output_dir, "knowledge_base.json"))
            print(f"   - '{query}': 找到 {len(results)} 个相关结果")
        
    except Exception as e:
        print(f"❌ 构建知识库时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
