# -*- coding: utf-8 -*-
import yaml
from typing import Dict, Any , Optional
from dataclasses import dataclass, field,asdict, InitVar
from configuration.logset import logger
from configuration.unit_config import LLMConfig, DatabaseConfig, RAGConfig




@dataclass
class GlobalConfig:
    """全局配置类，包含所有配置信息"""
    chat: LLMConfig
    embedding: LLMConfig
    rerank: LLMConfig
    vl: LLMConfig
    database: DatabaseConfig
    rag: RAGConfig  # 新增字段

    def to_dict(self) -> Dict[str, Any]:
        """将所有配置转换为字典格式"""
        return {
            "chat": self.chat.to_dict(),
            "embedding": self.embedding.to_dict(),
            "rerank": self.rerank.to_dict(),
            "vl": self.vl.to_dict(),
            "rag": self.rag.to_dict(),
            "database": {
                "neo4j": asdict(self.database.neo4j),
                "mongo": asdict(self.database.mongo),
                "milvus": asdict(self.database.milvus)
            },
        }
