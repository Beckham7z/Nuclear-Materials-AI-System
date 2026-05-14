import os
import re
import json
import base64
import numpy as np
import asyncio
from pathlib import Path
from typing import List, Dict, Any


from configuration.logset import logger
from configuration.global_config import GlobalConfig
from query.question_embedding import embedding_user_message, get_keyword4llm


class AbstractQueryEngine:
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
        # Milvus检索
        try:
            self.entities_vdb.update("collection_name", "abstract")
            vector = embedding[0] if isinstance(embedding, list) else embedding
            self.entities_vdb.client.load_collection("entity")
            self.entities_vdb.update("data", [vector])
            self.entities_vdb.update("top_k", top_k)
            return self.entities_vdb.milvus_query()[0]
        except Exception as e:
            logger.error(f"摘要检索失败: {str(e)}")
            return []
    
    def _get_entity_info(self, doi: str) -> Dict[str, Any]:
        # Neo4j查询摘要信息
        try:
            query = """
            MATCH (core:Abstract {doi: $doi})
            RETURN core
            """
            result = self.kg_client.neo4j_parameterized_query(query, {"doi": doi})
            result_data = result.get("node", [])
            if result_data:
                record = result_data[0]
                core_entity = record.get("core")
                doi = core_entity.get("doi", "")
                abstract = core_entity.get("abstract", "")
                return {
                    "doi": doi,
                    "abstract": abstract,
                }
            return {}
        except Exception as e:
            logger.error(f"实体信息查询失败: {str(e)}")
            return {}

    def _get_source_content(self, doi: str) -> List:
        # MongoDB查询源内容
        try:
            # self.source_context.update("collection_name", "full_docts")
            doc_id = "/home/shangqing/sqdata/mechdoi/"+doi+"/paper/auto/paper.md"
            param_dict = {"file_path": doi}
            doc_cursor = self.source_context.mongo_param_query(param_dict)
            doc_list = list(doc_cursor) if doc_cursor else []
            if not doc_list:
                logger.debug(f"未在集合 `{self.source_context.query_kword['collection_name']}` 中找到文档 _id={doc_id}")
                return []
            return doc_list
        except Exception as e:
            logger.error(f"Mongo检索失败: _id={doc_id}, 错误: {str(e)}")
            return []
        
    def _check_content_relationship(self, content: str) -> bool:
        """
        Check if the content is relevant to the question.
        :param content: Content to check
        :return: True if relevant, False otherwise
        """
        question = self.config.rag.Question.lower()
        content = content.lower()
        message = (
            f"Question: {question}\nContent: {content}\n"
            "Please determine whether the content is relevant to the question. "
            "Return a JSON object in the format: {\"related\": true/false}. Do not output anything else. "
            "If the content only contains references from journal articles, such as: "
            "[100] c. l. holloway, e. f. kuester, j. a. gordon, j. o’hara, j. booth, d. r. smith, ieee antenn. propag. m. 2012, 54, 10. "
            "[101] j. shi, z. li, d. k. sang, y. xiang, j. li, s. zhang, h. zhang, j. mater. chem. c 2018, 6, 1291. "
            "then return false."
        )

        for _ in range(3):
            self.chat_client.update_llmconfig("user_message", message)
            response = self.chat_client.get_response()

            match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            if not match:
                match = re.search(r"```\s*(\{.*?\})\s*```", response, re.DOTALL)
            if not match:
                match = re.search(r"(\{.*?\})", response, re.DOTALL)

            if match:
                try:
                    json_obj = json.loads(match.group(1))
                    if "related" in json_obj:
                        return bool(json_obj["related"])
                except Exception:
                    logger.error(f"Failed to parse JSON: {match.group(1)}")
                    pass
            logger.warning(f"Failed to extract valid JSON from LLM response: {response}")
            message = (
                "Please only return a JSON object in the format: {\"related\": true/false}, do not output anything else.\n"
                f"Question: {question}\nContent: {content}"
            )

        return False
    

    def _mergre_full_docx(self, top_k: int = 5) -> Dict[str, Dict]:
        """
        执行摘要查询，返回与问题相关的摘要信息列表。
        :param top_k: 返回的摘要数量
        :return: 包含摘要信息的字典列表
        """
        keywords = self._extract_keywords(self.config.rag.Question)
        logger.debug(f"提取到的关键词: {keywords}")
        self.chat_client.update_llmconfig("user_message", self.config.rag.Question)
        
        # 获取问题的向量表示
        embedding = self._get_query_embedding()
        if embedding is None:
            logger.warning("获取问题向量失败，可能是embedding服务不可用")
            return {}

        # 检索相关实体
        entities = self._retrieve_entities(embedding, top_k)
        if not entities:
            logger.warning("Milvus未检索到相关实体")
            return {}

        results = {}
        
        for abs_node in entities: 
            # 这里是Milvus的检索结果，因此用id
            logger.debug(f"检索到的实体: {abs_node.get('id', None)}")
            doi = abs_node.get("id", "")
            if not doi:
                logger.warning(f"检索到的实体没有DOI，跳过该实体{abs_node}")
                continue
            if doi not in results:
                results[doi] = {}
            # 从neo4j获取实体信息
            entity_info = self._get_entity_info(doi)
            if not entity_info:
                logger.warning(f"未能获取DOI {doi} 的摘要实体信息，跳过该实体")
                continue
            
            doi = entity_info.get("doi", "")
            abstract = entity_info.get("abstract", "")
            results[doi]["abstract"] = abstract
            
            # 获取源内容
            source_content_list = self._get_source_content(doi)
            logger.debug(f"检索到DOI {doi} 的源内容数量: {len(source_content_list)}")
            for source_content in source_content_list:
                if not source_content:
                    logger.warning(f"未能获取DOI {doi} 的源内容，跳过该实体")
                    continue
                index = source_content["chunk_order_index"]
                content = source_content["content"]
                logger.debug(f"检索到DOI {doi} 的源内容: {content[:50]}... (index: {index})")
                relationship = self._check_content_relationship(content)
                if relationship:
                    # 将结果添加到总结果中
                    results[doi][str(index)] = content
        return results
    
    def _format_doc_chunks(self, chunks: Dict[str, Dict[str, str]]) -> str:
        """
        chunks: {doi: {index: content, ...}, ...}
        """
        chunk_list = []
        for doi, doc_dict in chunks.items():
            for idx, content in doc_dict.items():
                if idx == "abstract":
                    continue
                chunk_list.append({"doi": doi, "index": idx, "content": content})
        chunks_json = json.dumps(chunk_list, ensure_ascii=False)
        return (
            "You are an academic assistant. Please carefully analyze the following document chunks. "
            "Whenever you cite a chunk, you must explicitly indicate its DOI (format: [DOI:xxx]). "
            "Do not fabricate any content not present in the prompt. Please answer the following question in a rigorous and academic manner:\n\n"
            f"Question: {self.config.rag.Question}\n\n"
            "-----Document Chunks (DC)-----\n\n"
            "Each chunk is structured as: {'doi': DOI, 'index': chunk number, 'content': content}\n"
            f"```json\n{chunks_json}\n```"
        )
    
    def query(self, top_k: int = 5) -> Dict[str, Any]:
        results = self._mergre_full_docx(top_k)
        if not results:
            logger.warning("未检索到相关摘要")
            return {"text": "未检索到相关摘要", "results": []}

        doi_summaries = {}
        for doi, doc_dict in results.items():
            # 取出所有文档段落（排除abstract）
            chunk_items = [(idx, content) for idx, content in doc_dict.items() if idx != "abstract"]
            group_size = 5
            group_summaries = []
            # 每5个为一组处理
            for i in range(0, len(chunk_items), group_size):
                group = chunk_items[i:i+group_size]
                group_json = json.dumps([
                    {"doi": doi, "index": idx, "content": content} for idx, content in group
                ], ensure_ascii=False)
                prompt = (
                    f"You are an academic assistant. Please carefully analyze the following document chunks. "
                    f"Whenever you cite a chunk, you must explicitly indicate its DOI (format: [DOI:{doi}]). "
                    f"Do not fabricate any content not present in the prompt. Please answer the following question in a rigorous and academic manner:\n\n"
                    f"Question: {self.config.rag.Question}\n\n"
                    f"-----Document Chunks (DC)-----\n\n"
                    f"```json\n{group_json}\n```"
                )
                self.chat_client.update_llmconfig("user_message", prompt)
                summary = self.chat_client.get_response()
                group_summaries.append(summary)
                self.history.append(prompt)
                self.history.append(summary)
            # 对该DOI下所有小结再做一次总结
            doi_summary_prompt = (
                f"You are an academic assistant. Please synthesize the following group answers for DOI {doi}, "
                f"and provide a professional summary. Be sure to retain DOI references (format: [DOI:{doi}]) in your conclusion:\n\n"
                + "\n\n".join(group_summaries)
            )
            self.chat_client.update_llmconfig("user_message", doi_summary_prompt)
            doi_summary = self.chat_client.get_response()
            doi_summaries[doi] = doi_summary
            self.history.append(doi_summary_prompt)
            self.history.append(doi_summary)

        # 合并所有DOI的总结，生成最终回答
        final_prompt = (
            "You are an academic assistant. Please synthesize the following summaries from different DOIs, "
            "ensuring that each conclusion retains DOI references. Provide a comprehensive academic answer:\n\n"
            + "\n\n".join([f"[DOI:{doi}]\n{summary}" for doi, summary in doi_summaries.items()])
        )
        self.chat_client.update_llmconfig("user_message", final_prompt)
        final_response = self.chat_client.get_response()
        self.history.append(final_prompt)
        self.history.append(final_response)

        return {
            "text": final_response,
            "results": results,
            "doi_summaries": doi_summaries
        }
    
    async def _summarize_group_async(self, doi, group, group_json):
        prompt = (
            f"You are an academic assistant. Please carefully analyze the following document chunks. "
            f"Whenever you cite a chunk, you must explicitly indicate its DOI (format: [DOI:{doi}]). "
            f"Do not fabricate any content not present in the prompt. Please answer the following question in a rigorous and academic manner:\n\n"
            f"Question: {self.config.rag.Question}\n\n"
            f"-----Document Chunks (DC)-----\n\n"
            f"```json\n{group_json}\n```"
        )
        return await self.chat_client.get_response("chat", user_message=prompt)

    async def query_async(self, top_k: int = 5) -> dict:
        results = self._mergre_full_docx(top_k)
        if not results:
            logger.warning("未检索到相关摘要")
            return {"text": "未检索到相关摘要", "results": []}
        doi_summaries = {}
        concurrency = getattr(self.config.rag, 'concurrency', 10)
        sem = asyncio.Semaphore(concurrency)
        async def summarize_doi(doi, doc_dict):
            chunk_items = [(idx, content) for idx, content in doc_dict.items() if idx != "abstract"]
            group_size = 5
            group_summaries = []
            # 每5个为一组处理
            for i in range(0, len(chunk_items), group_size):
                group = chunk_items[i:i+group_size]
                group_json = json.dumps([
                    {"doi": doi, "index": idx, "content": content} for idx, content in group
                ], ensure_ascii=False)
                summary = await self._summarize_group_async(doi, group, group_json)
                group_summaries.append(summary)
                self.history.append(group_json)
                self.history.append(summary)
            doi_summary_prompt = (
                f"You are an academic assistant. Please synthesize the following group answers for DOI {doi}, "
                f"and provide a professional summary. Be sure to retain DOI references (format: [DOI:{doi}]) in your conclusion:\n\n"
                + "\n\n".join(group_summaries)
            )
            doi_summary = await self.chat_client.get_response("chat", user_message=doi_summary_prompt)
            self.history.append(doi_summary_prompt)
            self.history.append(doi_summary)
            return doi, doi_summary
        tasks = [summarize_doi(doi, doc_dict) for doi, doc_dict in results.items()]
        doi_summary_pairs = await asyncio.gather(*tasks)
        for doi, summary in doi_summary_pairs:
            doi_summaries[doi] = summary
        final_prompt = (
            "You are an academic assistant. Please synthesize the following summaries from different DOIs, "
            "ensuring that each conclusion retains DOI references. Provide a comprehensive academic answer:\n\n"
            + "\n\n".join([f"[DOI:{doi}]\n{summary}" for doi, summary in doi_summaries.items()])
        )
        final_response = await self.chat_client.get_response("chat", user_message=final_prompt)
        self.history.append(final_prompt)
        self.history.append(final_response)
        return {
            "text": final_response,
            "results": results,
            "doi_summaries": doi_summaries
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        try:
            self.chat_client.update_llmconfig("user_message", query)
            return get_keyword4llm(self.chat_client)
        except Exception as e:
            logger.error(f"关键词提取失败: {str(e)}")
            return []

    def reset_history(self):
        self.history.clear()

