# -*- coding: utf-8 -*-
import json
from typing import List, Dict, Any
from configuration.logset import logger
from configuration.global_config import GlobalConfig
from query.question_embedding import embeddingALL
from query.rerank_enhance import rerank_enhance


"""Design structures with the largest negative Poisson's ratio"""

class ReserchProtoclosHandler():
    def __init__(self, global_config: GlobalConfig):
        # 初始化
        self.global_config = global_config
        self.milvus_client = self.global_config.database.milvus.client
        self.neo4j_client = self.global_config.database.neo4j.client
        self.mongo_client = self.global_config.database.mongo.client
        self.chat_client = self.global_config.chat.client
        self.embedding_client = self.global_config.embedding.client
        self.rerank_client = self.global_config.rerank.client
        self.rag_config = self.global_config.rag


    # 将问题和关键词向量化 这里只用调用一次，因此不使用并发
    async def _get_query_embedding(self) -> dict:
        try:
            return await embeddingALL(self.global_config, self.chat_client, self.embedding_client)
        except Exception as e:
            logger.error(f"调用 embedding_user_message 失败: {str(e)}")
            return {}
        
    # 在Milvus中查询向量
    def _retrieve_entities(self, embedding: Any) -> List[Dict[str, Any]]:
        try:
            self.milvus_client.update("top_k",self.global_config.rag.top_k)
            self.milvus_client.update("collection_name", "abstract")
            self.milvus_client.update("data", embedding)
            milvus_result = self.milvus_client.milvus_query()
            # 这里得到的是全部向量的列表，是否需要在这里直接检索相同向量？
            similar_vectors = [item for sublist in milvus_result for item in sublist]
            logger.debug(f"检索到相似向量{len(similar_vectors)}个")
            return similar_vectors
        except Exception as e:
            logger.error(f"Milvus实体检索失败: {str(e)}")
            return []
        
    # Neo4j
    def _get_entity_info(self, entity_uid_list: List[str]) -> List[Dict[str, Any]]:
        try:
            query = """
            MATCH (n:Abstract)
            WHERE n.doi IN $entity_uid_list
            RETURN n
            """
            return self.neo4j_client.neo4j_parameterized_query(query, {"entity_uid_list": entity_uid_list})["node"]
        except Exception as e:
            logger.error(f"Neo4j 查询失败: {str(e)}")
            return []

    def _rerank_results(self, results:list[str]):
        """
        对查询结果进行重排序，基于相似度得分。
        """
        # 按照相似度得分降序排序
        sorted_list = rerank_enhance(self.global_config , results)

        return sorted_list # 每个元素的评分
    

    async def _analyze_abstracts(self, input: str) -> List[str]:
        # 分析论文在专业领域的重要性
        prompt = """
    # You will receive a dictionary containing information about several research papers in the following format:
    {{
    "doi_1": "Abstract content 1",
    "doi_2": "Abstract content 2",
    ...
    }}

    Based on the user's question, carefully analyze the content of each abstract and determine which paper is most relevant and most capable of addressing the question.

    You need to follow these steps:

    1. Understand the user's question.
    2. Read and understand the core content of each abstract.
    3. Compare the relevance of each abstract to the question.
    4. Select the paper that is most relevant and most valuable for answering the question.
    5. Return ONLY the DOI of this selected paper in the exact format specified below.

    Please return the result in this exact JSON format:
    {{"DOI": "the_selected_doi"}}

    ## Example
    Input:
    Question: What are some effective transfer learning methods in natural language processing?
    {{
        "10.1109/ACCESS.2021.3051234": "This paper studies the application of deep learning in medical image segmentation...",
        "10.1016/j.artint.2022.103678": "This paper discusses transfer learning methods in natural language processing..."
    }}
    Output:
    {{"DOI": "10.1016/j.artint.2022.103678"}}

    ## True Data
    {input_text}
        """
        try:
            response = await self.chat_client.get_response("chat", user_message=prompt.format(input_text=input))
            cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
            logger.debug(f"cleaned_response: {cleaned_response}")
            
            if cleaned_response:
                try:
                    # 使用json.loads替代eval，更安全
                    result = json.loads(cleaned_response)
                    if isinstance(result, dict) and "DOI" in result:
                        return [result["DOI"]]
                    else:
                        logger.error(f"返回格式不正确: {result}")
                        return []
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败: {str(e)}, 原始响应: {cleaned_response}")
                    # 尝试从响应中提取DOI
                    if "DOI" in cleaned_response:
                        # 简单的字符串匹配提取DOI
                        import re
                        doi_pattern = r'"DOI"\s*:\s*"([^"]+)"'
                        match = re.search(doi_pattern, cleaned_response)
                        if match:
                            return [match.group(1)]
                    return []
            else:
                logger.error("LLM返回空响应")
                return []
                
        except Exception as e:
            logger.error(f"分析摘要失败: {str(e)}")
            return []



    async def query_async(self) -> dict:
        # 得到问题-关键词向量字典
        vector_dict = await self._get_query_embedding()
        # 解析字典，转化为向量列表
        if not vector_dict:
            logger.error("向量化结果为空，无法进行查询")
            return {}
        vectors_list = [vector for vector in vector_dict.values() if isinstance(vector, list) and len(vector) > 0]
        if not vectors_list:
            logger.error("向量列表为空，无法进行查询")
            return {}
        # 查询Milvus
        milvus_results = self._retrieve_entities(vectors_list)
        logger.debug(f"Milvus检索结果: {milvus_results}")
        # 上述代码得到一个嵌套列表，其中的每一个元素都是字典。
        if not milvus_results:
            logger.error("Milvus检索结果为空")
            return {}
        # 统计DOI出现次数
        doi_list = [item.get("id") for item in milvus_results if "id" in item]
        vector_count = {}
        for doi in doi_list:
            if doi in vector_count:
                vector_count[doi] += 1
            else:
                vector_count[doi] = 1
        logger.debug(f"DOI出现次数统计: {vector_count}")
        # 按照出现次数对DOI进行排序，返回前top_k个
        sorted_doi = sorted(vector_count.items(), key=lambda x: x[1], reverse=True)
        top_k_doi = [doi for doi, count in sorted_doi[:self.global_config.rag.top_k]]
        logger.debug(f"Top {self.global_config.rag.top_k} DOI: {top_k_doi}")

        entity_node_info = self._get_entity_info(top_k_doi) # 检索neo4j中的实体信息
        # 获取摘要列表
        abstract_list = [item['n'].get('abstract', '') for item in entity_node_info]
        logger.debug(f"检索到的摘要列表: {abstract_list}")
        scores = self._rerank_results(abstract_list) # 对摘要列表进行重排序，返回的是排序后的得分列表
        # 返回结果
        # scored_items = list(zip(entity_node_info, scores))
        # scored_items = list(zip(top_k_doi, abstract_list, scores))
        # #  按照打分重排序（降序）
        # scored_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
        # 不进行重排序
        scored_items = {doi: abstract for doi, abstract in zip(top_k_doi, abstract_list)}
        input_text = f"Question:{self.global_config.rag.Question} \n {scored_items}"
        finally_doi_list = await self._analyze_abstracts(input_text) # 分析摘要，返回最相关的DOI
        logger.debug(f"列表: {finally_doi_list}")
        finally_doi = finally_doi_list[0]
        logger.debug(f"最优DOI: {finally_doi}")

        
        
        




