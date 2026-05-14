# local_query/local_query_engine.py
import json
import streamlit as st
from typing import List, Dict, Any
from configuration.global_config import GlobalConfig
from configuration.logset import logger
from query.question_embedding import embedding_user_message
from query.rerank_enhance import rerank_enhance

class LocalQueryEngine:
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self.milvus_client = self.global_config.database.milvus.client
        self.neo4j_client = self.global_config.database.neo4j.client
        self.mongo_client = self.global_config.database.mongo.client
        self.chat_client = self.global_config.chat.client
        self.embedding_client = self.global_config.embedding.client
        self.rerank_client = self.global_config.rerank.client
    # embedding
    def _get_query_embedding(self) -> Any:
        try:
            return embedding_user_message(self.global_config, self.chat_client, self.embedding_client)
        except Exception as e:
            logger.error(f"调用 embedding_user_message 失败: {str(e)}")
            return None
        
    # Milvus
    def _retrieve_entities(self, embedding: Any) -> List[Dict[str, Any]]:
        try:
            self.milvus_client.update("top_k",self.global_config.rag.top_k)
            self.milvus_client.update("collection_name", "entity")
            self.milvus_client.update("data", embedding)
            milvus_result = self.milvus_client.milvus_query()
            similar_vectors = [item for sublist in milvus_result for item in sublist]
            logger.debug(f"检索到的相似向量{similar_vectors}")
            return similar_vectors
        except Exception as e:
            logger.error(f"Milvus实体检索失败: {str(e)}")
            return []
        
    # Neo4j
    def _get_entity_info(self, entity_uid_list: List[str]) -> List[Dict[str, Any]]:
        try:
            query = """
            MATCH (n:Entity)
            WHERE n.entity_uid IN $entity_uid_list
            RETURN n
            """
            return self.neo4j_client.neo4j_parameterized_query(query, {"entity_uid_list": entity_uid_list})["node"]
        except Exception as e:
            logger.error(f"Neo4j 查询失败: {str(e)}")
            return []
        
    # MongoDB
    def _get_source_content(self, doc_id: str):
        try:
            doc = self.mongo_client.mongo_id_query(doc_id)
            if not doc:
                logger.debug(f"未在集合 `{self.mongo_client.query_kword["collection_name"]}` 中找到文档 _id={doc_id}")
                return None
            if "content" not in doc or "file_path" not in doc:
                logger.debug(f"文档 _id={doc_id} 缺失字段: {doc}")
                return None
            return {doc["file_path"]: doc["content"]}
        except Exception as e:
            logger.error(f"Mongo检索失败: _id={doc_id}, 错误: {str(e)}")
            return None
        
    def _rerank_results(self, results:list[str]):
        """
        对查询结果进行重排序，基于相似度得分。
        """
        # 按照相似度得分降序排序
        sorted_list = rerank_enhance(self.global_config , results)

        return sorted_list # 每个元素的评分
    
    def _format_doc_chunks(self, chunks:Any) -> str:
        
        return f"""You are an academic assistant, please answer the following questions professionally and rigorously, using the citation format and clearly indicating the source of the DOI cited in your response, based on the content of the document paragraphs provided below:\n\nQuestion：{self.global_config.rag.Question}\n\n-----Document Chunks(DC)-----\n\n```json\n{chunks}\n```It is forbidden to appear in content that is not in the original text.The DOI is the key passed into the json dictionary,please indicate the source by adding [DOI] after the corresponding sentence."""
    
    def _get_final_response(self, results: List[Dict[str, Any]]) -> str:
        final_message = f"""Please carefully analyze the following document content and synthesize a comprehensive academic response.

        Question: {self.global_config.rag.Question}

        Based on the following document content:
        {chr(10).join(f'Document {i+1} (DOI: {r["doi"]}):\n{r["response"]}' for i, r in enumerate(results))}

        Your response should be structured as follows:
        1. **Introduction**: Briefly introduce the overall context and significance of the question.
        2. **Detailed Analysis of Each Document**: For each document, clearly and thoroughly summarize the key points, findings, and their academic implications. Please use academic language and ensure each document's contribution is explicitly discussed. When citing specific information from a document, please indicate the source by adding [DOI] after the corresponding sentence.
        3. **Conclusion**: Provide an integrated summary, highlighting the main insights and their relevance to the question.

        Please ensure your answer is rigorous, objective, and meets academic standards.
        """
        self.chat_client.update_llmconfig("user_message", final_message)
        self.chat_client.update_llmconfig("kwargs", {"temperature":0.7, "max_tokens": 8192,"stream":False})
        final_response = self.chat_client.get_response()
        return final_response

    def _NodeFirstQuery(self) -> Dict[str, Any]:
        embeddings = self._get_query_embedding()
        similar_vectors = self._retrieve_entities(embeddings)
        # 从 Milvus结果 中提取对应的 distance
        distances = [item.get('distance', None) for item in similar_vectors]
        entity_uid_list = [item.get("id", "") for item in similar_vectors]
        entity_node_info = self._get_entity_info(entity_uid_list)
        for item in entity_node_info:
            logger.debug(f"检索到的节点 {item['n']['entity_id']}")
        
        # 对实体节点进行重排序
        entity_id= [item['n'].get('entity_id', '') for item in entity_node_info]
        scores = self._rerank_results(entity_id)
        scored_items = list(zip(entity_node_info, scores, distances))
        sorted_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
        sorted_info = [
                {
                    "node": item['n'],
                    'distance': distance,
                    'rerank_score': score
                }
                for item, score, distance in sorted_items
            ]
        top_k_sorted_info = sorted_info[:self.global_config.rag.top_k] # 只保留前Top K个节点
        # logger.debug(f"排序后的实体节点信息: {top_k_sorted_info}")

        collected_docs = []# 初始化用于存储文档内容的列表
        for node_info in top_k_sorted_info:
            node = node_info["node"]
            description = node.get("description", "")
            source_id = node.get("source_id", "")
            # 按照<SEP>拆分为列表
            description_list = description.split("<SEP>")
            source_id_list = source_id.split("<SEP>")
            # 对应每个source_id获取对应的段落
            for src_id in  source_id_list:
                source_content = self._get_source_content(src_id)
                if source_content:
                    for doi, content in source_content.items():
                        message = self._format_doc_chunks({doi: content}) # 检索到的源文本
                        self.chat_client.update_llmconfig("kwargs", {"temperature":0.7, "max_tokens": 1024,"stream":False})
                        self.chat_client.update_llmconfig("user_message", message)
                        response = self.chat_client.get_response()
                        collected_docs.append({
                            "doi": doi,
                            "source_id": src_id,
                            "content": content,
                            "response": response
                        })
                else:
                    logger.warning(f"未获取到source_id={src_id}对应的段落内容")

            final_response = self._get_final_response(collected_docs)
            return final_response

    def Page_Display(self):
        final_response = self._NodeFirstQuery()
        st.markdown("### Comprehensive Analysis")
        st.markdown("---")
        st.markdown(final_response)  # 最后一个元素是综合分析的回答
            


