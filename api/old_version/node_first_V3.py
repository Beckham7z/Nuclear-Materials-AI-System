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
        self.source_context = global_config.database.mongo.client
        self.history: List[str] = []

    def _get_query_embedding(self) -> Any:
        try:
            return embedding_user_message(self.config, self.chat_client, self.embedding_client)
        except Exception as e:
            logger.error(f"调用 embedding_user_message 失败: {str(e)}")
            return None

    def _retrieve_entities(self, embedding: Any, top_k: int = 5) -> List[Dict[str, Any]]:
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
            RETURN core
            """
            result = self.kg_client.neo4j_parameterized_query(query, {"entity_uid": entity_uid})
            result_data = result.get("node", [])
            if result_data:
                record = result_data[0]
                core_entity = record.get("core")
                source_id = core_entity.get("source_id", "")
                description = core_entity.get("description", "")
                return {
                    "id": core_entity.get("entity_uid", entity_uid),
                    "name": core_entity.get("entity_id", ""),
                    "description": description.split("<SEP>") if "<SEP>" in description else [description],
                    "source_id": source_id.split("<SEP>") if "<SEP>" in source_id else [source_id],
                    "file_path": core_entity.get("file_path", ""),
                }
            return {}
        except Exception as e:
            logger.error(f"实体信息查询失败: {str(e)}")
            return {}
        
    def _get_source_content(self, doc_id: str):
        try:
            doc = self.source_context.mongo_id_query(doc_id)
            if not doc:
                logger.debug(f"未在集合 `{self.source_context.query_kword["collection_name"]}` 中找到文档 _id={doc_id}")
                return None
            if "content" not in doc or "file_path" not in doc:
                logger.debug(f"文档 _id={doc_id} 缺失字段: {doc}")
                return None
            return {doc["file_path"]: doc["content"]}
        except Exception as e:
            logger.error(f"Mongo检索失败: _id={doc_id}, 错误: {str(e)}")
            return None
    
    def query_mongo(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        keywords = self._extract_keywords(user_query)
        logger.debug(f"提取到的关键词: {keywords}")
        if not keywords:
            return {"text": "未提取到关键词", "results": []}

        embedding = self._get_query_embedding()
        if not embedding:
            return {"text": "向量生成失败", "results": []}

        candidates = self._retrieve_entities(embedding, top_k)
        logger.debug(f"检索到的实体候选: {candidates}")
        if not candidates:
            return {"text": "未检索到相关实体", "results": []}

        collected_docs = {}
        for item in candidates:
            entity_uid = item.get("id")
            if not entity_uid:
                logger.error(f"检索到的实体{item}没有ID，跳过该实体")
                continue
            entity_info = self._get_entity_info(entity_uid)
            source_ids = entity_info["source_id"] if entity_info else []
            logger.debug(f"检索到实体 {entity_info.get('name', entity_uid)} 的源ID: {source_ids}")
            for sid in source_ids:
                content_pair = self._get_source_content(sid)
                logger.debug(f"MongoDB中检索到文档 {content_pair} 的内容")
                if content_pair:
                    collected_docs.update(content_pair)

        context = self._format_doc_chunks(collected_docs, self.config.rag.Question)
        full_prompt = "\n".join(self.history + [context])

        try:
            self.chat_client.update_llmconfig("user_message", full_prompt)
            response = self.chat_client.get_response()
            self.history.append(context)
            self.history.append(response)

            if len(collected_docs) > 5:
                summary_prompt = f"你是一个学术助理。请对以下多个回答进行归纳总结，输出一段专业性的总结：\n\n问题：{self.config.rag.Question}\n\n{response}"
                self.chat_client.update_llmconfig("user_message", summary_prompt)
                response = self.chat_client.get_response()
                self.history.append(summary_prompt)
                self.history.append(response)

            return {"text": response, "results": collected_docs}
        except Exception as e:
            logger.error(f"LLM回答失败: {str(e)}")
            return {"text": "LLM生成回答失败", "results": []}

    def _format_doc_chunks(self, chunks: Dict[str, str], question: str) -> str:
        chunks_json = json.dumps([{"doi": k, "content": v} for k, v in chunks.items()], ensure_ascii=False)
        return f"""你是一个学术助理，请根据下列提供的文档段落内容，使用引用格式并在回答中明确指出所引用的DOI来源，专业严谨地回答如下问题：\n\n问题：{question}\n\n-----Document Chunks(DC)-----\n\n```json\n{chunks_json}\n```"""

    def _extract_keywords(self, query: str) -> List[str]:
        try:
            self.chat_client.update_llmconfig("user_message", query)
            return get_keyword4llm(self.chat_client)
        except Exception as e:
            logger.error(f"关键词提取失败: {str(e)}")
            return []

    def reset_history(self):
        self.history.clear()
