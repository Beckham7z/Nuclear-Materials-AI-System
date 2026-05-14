"""
RAG 检索模块
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import yaml
import asyncio

from src.configuration.logset import logger
from src.llm.async_utils import run_async

# LightRAG 相关
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc
    LIGHTRAG_AVAILABLE = True
except ImportError:
    LIGHTRAG_AVAILABLE = False
    logger.warning("LightRAG 未安装，将使用简化检索")


class RAGClient:
    """RAG 客户端封装"""
    
    def __init__(self, config_path: str = None):
        """初始化 RAG 客户端"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        self.config = self._load_config(config_path)
        self.rag_config = self.config.get('rag', {})
        self.embedding_config = self.config.get('embedding', {})
        
        self.working_dir = self.rag_config.get('working_dir', '/home/zyx/A_project/mechpy_nuc_mat/myKG')
        self.top_k = self.rag_config.get('top_k', 10)
        self.enable_rerank = self.rag_config.get('enable_rerank', False)
        
        self.rag = None
        self._initialized = False
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}")
            return {}
    
    async def initialize(self):
        """异步初始化 RAG"""
        if self._initialized or not LIGHTRAG_AVAILABLE:
            return
        
        try:
            embed_model = self.embedding_config.get('model', 'bge-m3:latest')
            host = self.embedding_config.get('host', 'http://127.0.0.1:11434')
            
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=ollama_model_complete,
                llm_model_name=self.rag_config.get('llm_model', 'qwen3:30b'),
                llm_model_kwargs={
                    "host": host,
                    "options": {"num_ctx": 32768},
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.embedding_config.get('dimension', 1024),
                    max_token_size=8192,
                    func=lambda texts: ollama_embed(
                        texts=texts, 
                        embed_model=embed_model, 
                        host=host
                    ),
                ),
                kv_storage="JsonKVStorage",
                graph_storage="NetworkXStorage",
                vector_storage="NanoVectorDBStorage",
            )
            
            await self.rag.initialize_storages()
            self._initialized = True
            logger.info("RAG 初始化成功")
            
        except Exception as e:
            logger.error(f"RAG 初始化失败: {e}")
            self._initialized = False
    
    def initialize_sync(self):
        """同步初始化 RAG"""
        if self._initialized or not LIGHTRAG_AVAILABLE:
            return
        
        try:
            # 使用 asyncio 运行异步初始化
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.initialize())
        except Exception as e:
            logger.error(f"RAG 同步初始化失败: {e}")
    
    async def query(self, query_text: str, mode: str = "hybrid", top_k: int = None) -> Dict[str, Any]:
        """执行 RAG 查询"""
        if not self._initialized:
            await self.initialize()
        
        if self.rag is None:
            return {
                "context": "RAG 未初始化",
                "results": [],
                "mode": mode
            }
        
        k = top_k or self.top_k
        
        try:
            param = QueryParam(
                mode=mode,
                top_k=k,
                enable_rerank=self.enable_rerank,
                only_need_context=True
            )
            
            context = await self.rag.aquery(query_text, param=param)
            
            # 解析结果
            results = self._parse_context(context)
            
            return {
                "context": context,
                "results": results,
                "mode": mode,
                "query": query_text,
                "top_k": k
            }
            
        except Exception as e:
            logger.error(f"RAG 查询失败: {e}")
            return {
                "context": "",
                "results": [],
                "error": str(e),
                "mode": mode
            }
    
    def query_sync(self, query_text: str, mode: str = "hybrid", top_k: int = None) -> Dict[str, Any]:
        """同步 RAG 查询"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.query(query_text, mode, top_k))
    
    def _parse_context(self, context: str) -> List[Dict[str, Any]]:
        """解析 RAG 上下文"""
        results = []
        
        if not context or not isinstance(context, str):
            return results
        
        try:
            chunks = [c.strip() for c in context.split("\n\n") if c.strip()]
            
            content_mapping = [
                {"index": 1, "type": "entity", "title": "知识图谱实体数据", "base_score": 0.95},
                {"index": 3, "type": "relationship", "title": "知识图谱关系数据", "base_score": 0.92},
                {"index": 5, "type": "document", "title": "文献文档片段", "base_score": 0.88}
            ]
            
            for mapping in content_mapping:
                if mapping["index"] < len(chunks):
                    chunk = chunks[mapping["index"]]
                    content_score = self._calculate_content_relevance(chunk, "")
                    
                    results.append({
                        "text": chunk,
                        "score": round(mapping["base_score"] * content_score, 4),
                        "type": mapping["type"],
                        "title": mapping["title"],
                        "metadata": {
                            "content_length": len(chunk),
                            "has_json": chunk.count('{') > 0,
                        }
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            logger.error(f"解析上下文失败: {e}")
        
        return results
    
    def _calculate_content_relevance(self, content: str, query: str) -> float:
        """计算内容相关性"""
        score = 1.0
        
        if len(content) > 1000:
            score *= 1.1
        elif len(content) < 100:
            score *= 0.8
        
        if content.count('{') > 5:
            score *= 1.05
        
        return min(max(score, 0.8), 1.2)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取 RAG 统计信息"""
        if not self._initialized:
            return {"initialized": False}
        
        try:
            # 尝试获取存储统计
            stats = {
                "initialized": True,
                "working_dir": self.working_dir,
                "top_k": self.top_k,
                "enable_rerank": self.enable_rerank
            }
            
            if self.rag:
                # 可以添加更多统计信息
                pass
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"initialized": False, "error": str(e)}


# 全局 RAG 客户端实例
_global_rag_client = None


def get_rag_client(config_path: str = None) -> RAGClient:
    """获取全局 RAG 客户端"""
    global _global_rag_client
    if _global_rag_client is None:
        _global_rag_client = RAGClient(config_path)
    return _global_rag_client
