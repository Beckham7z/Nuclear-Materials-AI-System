import csv
import time
import json
from transformers import AutoTokenizer
from typing import List, Callable, Optional, Dict, Any
import math
import re
import os
import asyncio
import nest_asyncio

nest_asyncio.apply()

# WorkingDir - 调整为您的项目路径
ROOT_DIR = "/home/beckham7/A_project/mechpy"  # 修改为您的项目根目录
WORKING_DIR = os.path.join(ROOT_DIR, "myKG")
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")

# 数据库配置保持不变
os.environ["MONGO_URI"] = "mongodb://root:root@localhost:27017/"
os.environ["MONGO_DATABASE"] = "mech"

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "1q2w3e4r"

os.environ["MILVUS_URI"] = "http://localhost:19530"
os.environ["MILVUS_USER"] = "root"
os.environ["MILVUS_PASSWORD"] = "root"
os.environ["MILVUS_DB_NAME"] = "mech"


class MarkdownChunker:
    def __init__(
        self,
        file_path: str,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        max_token: Optional[int] = None,
        merge_all: bool = False,
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_token = max_token or float("inf")
        self.merge_all = merge_all
        self.exclusion_keywords = [
            'code availability', 'data availability', 'references',
            'supplementary materials', 'acknowledgements',
            'conflicts of interest', 'author contributions',
            'acknowledgments', 'declaration of competing interest',
            'credit authorship contribution statement', "orcid ids", "funding",
            "Declaration of conflicting interests"
        ]

    def _read_file(self) -> str:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _split_by_headers(self, content: str) -> List[str]:
        blocks = re.split(r'(?=^#{1,6} )', content, flags=re.MULTILINE)
        return [block.strip() for block in blocks if block.strip()]

    def _filter_blocks(self, blocks: List[str]) -> List[str]:
        filtered = []
        for block in blocks:
            if not any(keyword.lower() in block.lower().splitlines()[0] for keyword in self.exclusion_keywords):
                filtered.append(block)
        return filtered

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer(text))
        return 0

    def _chunk_large_block(self, block: str) -> List[Dict[str, Any]]:
        if not self.tokenizer:
            return [{"content": block, "token_num": 0}]

        tokens = self.tokenizer(block)
        token_len = len(tokens)
        num_chunks = math.ceil(token_len / self.max_token)
        chunk_size = math.ceil(token_len / num_chunks)
        chunks = []

        for i in range(num_chunks):
            start = max(0, i * chunk_size - 10)
            end = min(len(tokens), (i + 1) * chunk_size + 10)
            chunk_tokens = tokens[start:end]
            chunk_text = self._decode_tokens(chunk_tokens)
            chunks.append(
                {"content": chunk_text, "token_num": len(chunk_tokens)})
        return chunks

    def _decode_tokens(self, tokens: List[str]) -> str:
        if self.tokenizer:
            if hasattr(self.tokenizer, "decode"):
                return self.tokenizer.decode(tokens)
            elif hasattr(self.tokenizer, "convert_tokens_to_string"):
                return self.tokenizer.convert_tokens_to_string(tokens)
            else:
                return "".join(tokens)
        return "".join(tokens)

    def _merge_blocks(self, blocks: List[str]) -> List[Dict[str, Any]]:
        if not self.tokenizer:
            return [{"content": "\n\n".join(blocks), "token_num": 0}]

        result = []
        current_block = ""
        current_token_num = 0

        for block in blocks:
            token_num = self._count_tokens(block)
            if token_num > self.max_token:
                if current_block:
                    result.append({
                        "content": current_block.strip(),
                        "token_num": current_token_num
                    })
                    current_block = ""
                    current_token_num = 0
                result.extend(self._chunk_large_block(block))
                continue

            if current_token_num + token_num <= self.max_token:
                current_block += "\n\n" + block
                current_token_num += token_num
            else:
                result.append({
                    "content": current_block.strip(),
                    "token_num": current_token_num
                })
                current_block = block
                current_token_num = token_num

        if current_block:
            result.append({
                "content": current_block.strip(),
                "token_num": current_token_num
            })

        return result

    def process(self) -> List[Dict[str, Any]]:
        raw_content = self._read_file()
        blocks = self._split_by_headers(raw_content)
        blocks = self._filter_blocks(blocks)

        if self.merge_all:
            assert self.max_token == float(
                "inf"), "max_token must be infinity when merging all"
            full_text = "\n\n".join(blocks)
            return [{
                "content": full_text,
                "token_num": self._count_tokens(full_text)
            }]

        return self._merge_blocks(blocks)


def create_simple_knowledge_base():
    """创建简化的知识库，与智能系统兼容"""
    
    md_file_path = "/home/beckham7/A_project/n_material_file/converted_output/converted.md"
    output_dir = "/home/beckham7/A_project/n_material_file/knowledge_base"
    
    # 检查文件是否存在
    if not os.path.exists(md_file_path):
        print(f"错误: 文件不存在 - {md_file_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"开始处理: {md_file_path}")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        
        # 读取Markdown文件
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"文档内容长度: {len(content)} 字符")
        
        # 使用MarkdownChunker处理文档
        chunker = MarkdownChunker(md_file_path, merge_all=False)
        raw_chunks = chunker.process()
        
        # 为每个chunk添加id字段
        chunks = []
        for i, chunk in enumerate(raw_chunks):
            chunk_with_id = chunk.copy()
            chunk_with_id['id'] = i
            chunks.append(chunk_with_id)
        
        print(f"分割为 {len(chunks)} 个文档块")
        
        # 创建知识库结构
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
        
        print(f"知识库已保存: {kb_file}")
        
        # 创建搜索索引
        create_search_index(chunks, output_dir)
        
        print(f"处理完成: {md_file_path}")
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        
        return knowledge_base
        
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        import traceback
        traceback.print_exc()


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
    
    print(f"搜索索引已创建: {index_file}")
    print(f"索引包含 {len(keyword_index)} 个关键词")


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


def test_chunker():
    """测试 MarkdownChunker 功能"""
    test_file_path = "/home/beckham7/A_project/n_material_file/converted_output/converted.md"
    
    if not os.path.exists(test_file_path):
        print(f"测试文件不存在: {test_file_path}")
        return
    
    chunker = MarkdownChunker(test_file_path, merge_all=True)
    chunks = chunker.process()
    
    print(f"文档块数量: {len(chunks)}")
    if chunks:
        print(f"第一块内容预览: {chunks[0]['content'][:500]}...")


def main():
    """主函数 - 构建与智能系统兼容的知识库"""
    
    print("=" * 60)
    print("🧠 知识库构建器 - 智能系统兼容版本")
    print("=" * 60)
    
    # 测试 chunker 功能
    print("=== 测试 MarkdownChunker ===")
    test_chunker()
    
    print("\n=== 开始构建知识库 ===")
    # 创建简化的知识库
    kb = create_simple_knowledge_base()
    
    if kb:
        print("\n" + "=" * 60)
        print("✅ 知识库构建完成!")
        print("=" * 60)
        print(f"📊 统计信息:")
        print(f"   - 源文件: {kb['source_file']}")
        print(f"   - 创建时间: {kb['created_time']}")
        print(f"   - 文档块数量: {kb['total_chunks']}")
        print(f"   - 输出目录: /home/beckham7/A_project/n_material_file/knowledge_base")
        
        # 测试搜索功能
        print(f"\n🔍 测试搜索功能:")
        test_queries = [
            "nanocrystalline grains",
            "Xe irradiation",
            "hardness measurement"
        ]
        
        for query in test_queries:
            results = search_knowledge_base(query, "/home/beckham7/A_project/n_material_file/knowledge_base/knowledge_base.json")
            print(f"   - '{query}': 找到 {len(results)} 个相关结果")


if __name__ == "__main__":
    main()
