# local_query/local_query_engine.py
import json
import numpy as np
from typing import List, Dict, Any
from configuration.global_config import GlobalConfig
from configuration.logset import logger
from query.question_embedding import embedding_user_message, get_keyword4llm

class LocalQueryEngine:
    def __init__(self, global_config: GlobalConfig):
        self.config = global_config
        self.chat_client = global_config.chat.client
        self.embedding_client = global_config.embedding.client
        self.entities_vdb = global_config.database.milvus.client
        self.kg_client = global_config.database.neo4j.client
        self.mongo_client = global_config.database.mongo.client
        self.history: List[str] = []

    def _get_query_embedding(self) -> Any:
        try:
            return embedding_user_message(self.config, self.chat_client, self.embedding_client)
        except Exception as e:
            logger.error(f"调用 embedding_user_message 失败: {str(e)}")
            return None

    def _retrieve_entities(self, embedding: Any, top_k: int = 50) -> List[Dict[str, Any]]:
        try:
            self.entities_vdb.update("collection_name", "entity")
            vector = embedding[0] if isinstance(embedding, list) else embedding
            self.entities_vdb.client.load_collection("entity")
            self.entities_vdb.update("data", [vector])
            self.entities_vdb.update("top_k", top_k)
            return self.entities_vdb.milvus_query()[0]
        except Exception as e:
            logger.error(f"实体检索失败: {str(e)}")
            return []

    def _get_entity_info(self, entity_uid: str) -> Dict[str, Any]:
        try:
            query = """
            MATCH (core:Entity {entity_uid: $entity_uid})
            OPTIONAL MATCH (core)-[:DIRECT*1..1]-(linked:Entity)
            RETURN core, collect(DISTINCT linked) AS entities
            """
            result = self.kg_client.neo4j_parameterized_query(query, {"entity_uid": entity_uid})
            result_data = result.get("node", [])
            if result_data:
                record = result_data[0]
                core_entity = record.get("core")
                entities = [e for e in record.get("entities", []) if e is not None]
                source_id = core_entity.get("source_id", "")
                description = core_entity.get("description", "")
                return {
                    "id": core_entity.get("entity_uid", entity_uid),
                    "name": core_entity.get("entity_id", ""),
                    "description": description.split("<SEP>") if "<SEP>" in description else [description],
                    "source_id": source_id.split("<SEP>") if "<SEP>" in source_id else [source_id],
                    "file_path": core_entity.get("file_path", ""),
                    "connected_entities": [e.get("entity_id", "") for e in entities if isinstance(e, dict)]
                }
            return {}
        except Exception as e:
            logger.error(f"实体信息查询失败: {str(e)}")
            return {}

    def _get_source_content(self, doc_id: str) -> str:
        try:
            collection = self.source_context[self.source_context.query_kword["collection_name"]]
            doc = collection.find_one({"_id": doc_id}, {"content": 1, "_id": 0})
            return doc.get("content") if doc else None
        except Exception as e:
            logger.error(f"Mongo检索失败: {str(e)}")
            return None

    def _expand_knowledge_context(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched = []
        for item in entities:
            entity_uid = item.get("id")
            detail = self._get_entity_info(entity_uid)
            if detail:
                enriched.append(detail)
        return enriched

    def _format_for_llm_neo4j(self, entities: List[Dict[str, Any]]) -> str:
        question = self.config.rag.Question
        entities_str = json.dumps(entities, ensure_ascii=False)
        return f"""你是一个力学超材料领域学术助理，请根据以下实体信息和它们之间的关系，使用专业术语、准确清晰的语言回答用户问题：\n\n问题：{question}\n\n-----Entities(KG)-----\n\n```json\n{entities_str}\n```\n\n-----Relationships(KG)-----\n\n```json\n[]\n```\n\n-----Document Chunks(DC)-----\n\n```json\n[]\n```"""

    def _extract_keywords(self, query: str) -> List[str]:
        try:
            self.chat_client.update_llmconfig("user_message", query)
            return get_keyword4llm(self.chat_client)
        except Exception as e:
            logger.error(f"关键词提取失败: {str(e)}")
            return []

    def query_neo4j(self, user_query: str, top_k: int = 50, max_entities: int = 20) -> Dict[str, Any]:
        keywords = self._extract_keywords(user_query)
        if not keywords:
            return {"text": "未提取到关键词", "results": []}

        query_text = " ".join(keywords)
        embedding = self._get_query_embedding()
        if not embedding:
            return {"text": "向量生成失败", "results": []}

        candidates = self._retrieve_entities(embedding, top_k)
        if not candidates:
            return {"text": "未检索到相关实体", "results": []}

        expanded = self._expand_knowledge_context(candidates[:max_entities])
        context = self._format_for_llm(expanded)

        full_prompt = "\n".join(self.history + [context])

        try:
            self.chat_client.update_llmconfig("user_message", full_prompt)
            response = self.chat_client.get_response()
            self.history.append(context)
            self.history.append(response)
            return {
                "text": response,
                "results": expanded
            }
        except Exception as e:
            logger.error(f"LLM回答失败: {str(e)}")
            return {"text": "LLM生成回答失败", "results": []}

    def reset_history(self):
        self.history.clear()
