# -*- coding: utf-8 -*-
import asyncio
import json
from typing import List, Dict, Any
from configuration.logset import logger
from configuration.global_config import GlobalConfig
from query.question_embedding import embeddingALL
from query.rerank_enhance import rerank_enhance
from prompt.ResearchScheme import ResearchSchemePrompt

"""Design structures with the largest negative Poisson's ratio"""

class ReserchPlanHandler():
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

    # MongoDB
    def _get_source_content(self, doc_id: str):
        try:
            self.mongo_client.update("collection_name", "full_docts")
            results = self.mongo_client.mongo_ids_query(doc_id)
            return results
        except Exception as e:
            logger.error(f"Mongo检索失败: _id={doc_id}, 错误: {str(e)}")
            return None

    async def _rerank_results(self, results:list[str]):
        """
        对查询结果进行重排序，基于相似度得分。
        """
        # 按照相似度得分降序排序
        sorted_list = await rerank_enhance(self.global_config , results)

        return sorted_list # 每个元素的评分
    

    async def _analyze_papers(self, input_txt: str) -> List[str]:
        # 分析并输出论文研究方案
        try:
            response = await self.chat_client.get_response("chat", user_message=input_txt, prompt=
ResearchSchemePrompt)
            logger.debug(f"response: {response}")
            return response  # 添加返回语句
                
        except Exception as e:
            logger.error(f"分析文章失败: {str(e)}")
            return ""

    async def _generate_comprehensive_research_plan(self, response_list: List[Dict[str, str]]) -> str:
        """
        基于多篇文章的分析结果，生成综合研究方案
        """
        # 构建输入文本，包含所有DOI和对应的分析结果
        papers_content = ""
        for i, response_dict in enumerate(response_list):
            for doi, analysis in response_dict.items():
                papers_content += f"\n\nPaper {i+1} [DOI: {doi}]:\n{analysis}"
        
                comprehensive_prompt = f"""
You are a research expert. Based on the following individual paper analyses, please provide a comprehensive and detailed research plan to address the question: "{self.global_config.rag.Question}"

Paper analyses:
{papers_content}

Provide a structured research plan with:

## Research Overview
Summarize current research state and key challenges.

## Research Objectives
Define specific, measurable research goals based on paper insights.

## Key Performance Indicators
Identify critical metrics for problem-solving success. Present as markdown table with complete data from ALL papers:

| Indicator | Target Value | Measurement Method | Source DOI | Additional Parameters |
|-----------|--------------|-------------------|------------|----------------------|
| [metric name] | [target] | [method] | [DOI: xxx] | [complete parameter set] |

**IMPORTANT**: Include ALL numerical data, experimental results, and performance metrics from EVERY paper. Do not omit any tables or data points from the original analyses.

## Methodology
Detail research methods and experimental approaches from papers, avoiding repetition. Include:
- Complete experimental setups and parameters from each paper
- All material properties and specifications mentioned
- Comprehensive fabrication methods and conditions
- Full characterization techniques and measurement protocols

## Implementation Steps
1. **Phase 1**: [tasks with specific parameters]
2. **Phase 2**: [tasks with detailed methodologies]
3. **Phase 3**: [tasks with validation approaches]

## Expected Outcomes
Describe anticipated results and potential breakthroughs with quantitative targets.

## Risk Assessment
- **Technical risks**: [mitigation strategies with specific approaches]
- **Resource risks**: [mitigation strategies with backup plans]

## Innovation Points
Highlight novel aspects distinguishing this research.

**Requirements:**
- Reference papers using [DOI: paper_doi] format
- Journal-style writing: concise, clear, no redundancy
- Base content strictly on provided literature
- Preserve ALL numerical data accuracy from original papers
- Include COMPLETE tables and datasets from each paper
- Integrate insights cohesively across multiple papers
- When multiple papers contain similar data, present ALL variations with proper attribution
- Ensure no data loss or omission from the original analyses
        """
        
        try:
            comprehensive_response = await self.chat_client.get_response(
                "chat", 
                user_message=comprehensive_prompt,
                prompt="You are an expert research consultant with deep knowledge in academic research methodology and scientific innovation."
            )
            logger.debug(f"生成综合研究方案，长度: {len(comprehensive_response)}")
            
            # 清除markdown代码块标记
            if comprehensive_response:
                # 移除开头和结尾的markdown标记
                cleaned_response = comprehensive_response.strip()
                if cleaned_response.startswith("```markdown"):
                    cleaned_response = cleaned_response[11:].strip()
                elif cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:].strip()
                
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3].strip()
                
                return cleaned_response
            
            return comprehensive_response
            
        except Exception as e:
            logger.error(f"生成综合研究方案失败: {str(e)}")
            # 如果生成失败，返回简单的汇总
            fallback_summary = f"无相关内容"
            return fallback_summary

    def _doi2link(self,text:str)-> str:
        """
        将DOI转换为链接格式
        """
        if not text:
            return text
        
        # 使用正则表达式匹配DOI并转换为链接
        import re
        doi_pattern = r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b'
        
        def replace_doi(match):
            doi = match.group(0)
            return f"[{doi}](https://doi.org/{doi})"
        
        return re.sub(doi_pattern, replace_doi, text, flags=re.IGNORECASE)

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
        top_k_doi_path = [f"/home/shangqing/sqdata/mechdoi/{doi}/paper/auto/paper.md" for doi in top_k_doi]
        # for doi_path in top_k_doi_path:
        #     full_texts = self._get_source_content(top_k_doi_path)        
        full_texts = self._get_source_content(top_k_doi_path)
        
        if not full_texts:
            logger.error("未获取到任何全文内容")
            return {}
        
        # 获取并发数配置
        concurrency = self.rag_config.concurrency
        
        # 处理full_texts，无论是列表还是字典
        if isinstance(full_texts, list):
            # 如果是列表，需要和DOI配对
            text_items = [(top_k_doi[i] if i < len(top_k_doi) else f"unknown_{i}", content) 
                         for i, content in enumerate(full_texts)]
            logger.debug(f"full_texts是列表格式，包含 {len(text_items)} 个文档")
        elif isinstance(full_texts, dict):
            text_items = list(full_texts.items())  # [(doi_path, content), ...]
            logger.debug(f"full_texts是字典格式，包含 {len(text_items)} 个文档")
        else:
            logger.error(f"full_texts格式{type(full_texts)}不正确，应为列表或字典")
            return {}
        
        # 根据并发数决定处理方式
        all_responses = []
        
        if len(text_items) <= concurrency:
            # 一次性全部处理
            logger.debug(f"文档数量 {len(text_items)} <= 并发数 {concurrency}，一次性处理")
            
            # 创建并发任务
            tasks = []
            for identifier, content in text_items:
                input_text = f"Question: {self.global_config.rag.Question}\n\nDocument Content:\n{content}"
                task = self._analyze_papers(input_text)
                tasks.append(task)
            
            # 并发执行所有任务
            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for i, response in enumerate(responses):
                    if isinstance(response, Exception):
                        logger.error(f"分析第 {i+1} 个文档时出错: {str(response)}")
                    else:
                        all_responses.extend(response if isinstance(response, list) else [response])
            except Exception as e:
                logger.error(f"并发执行任务时出错: {str(e)}")
                
        else:
            # 分批处理
            logger.debug(f"文档数量 {len(text_items)} > 并发数 {concurrency}，分批处理")
            
            # 将文档分批
            for i in range(0, len(text_items), concurrency):
                batch = text_items[i:i + concurrency]
                logger.debug(f"处理第 {i//concurrency + 1} 批，包含 {len(batch)} 个文档")
                
                # 创建当前批次的任务
                tasks = []
                for identifier, content in batch:
                    input_text = f"Question: {self.global_config.rag.Question}\n\nDocument Content:\n{content}"
                    task = self._analyze_papers(input_text)
                    tasks.append(task)
                
                # 并发执行当前批次的任务
                try:
                    batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
                    for j, response in enumerate(batch_responses):
                        if isinstance(response, Exception):
                            logger.error(f"分析第 {i+j+1} 个文档时出错: {str(response)}")
                        else:
                            all_responses.extend(response if isinstance(response, list) else [response])
                except Exception as e:
                    logger.error(f"处理第 {i//concurrency + 1} 批时出错: {str(e)}")
        
        # 处理分析结果
        logger.debug(f"共获得 {len(all_responses)} 个分析结果")
        
        # 添加调试信息，查看响应内容
        for i, response in enumerate(all_responses):
            logger.debug(f"Response {i}: type={type(response)}, content='{str(response)[:100]}...'")
        
        # 将响应与对应的DOI配对，存储为列表嵌套字典
        response_list = []
        for i, response in enumerate(all_responses):
            # 检查响应的类型和内容
            if isinstance(response, str) and response and response.strip():  # 过滤有效响应
                # 获取对应的DOI
                doi = text_items[i][0] if i < len(text_items) else f'unknown_{i}'
                response_dict = {doi: response.strip()}
                response_list.append(response_dict)
                logger.debug(f"添加有效响应 {i}: DOI={doi}")
            else:
                logger.debug(f"跳过无效响应 {i}: type={type(response)}, empty={not response}, content='{str(response)[:50]}'")
        
        if not response_list:
            logger.warning("未获得有效的分析结果，返回原始摘要信息")
            # 返回实体信息作为备选
            if entity_node_info:
                backup_list = []
                for i, info in enumerate(entity_node_info):
                    doi = info.get('doi', f'unknown_{i}')
                    abstract = info.get('abstract', '')
                    backup_list.append({doi: abstract})
                return backup_list
            return []
        
        logger.debug(f"成功处理 {len(response_list)} 个有效响应")

        # 将所有分析结果合并，再次输入LLM生成综合研究方案
        comprehensive_analysis = await self._generate_comprehensive_research_plan(response_list)
        

        comprehensive_analysis = self._doi2link(comprehensive_analysis)
        # 返回最终的综合研究方案
        return {
            "comprehensive_plan": comprehensive_analysis,
            "individual_analyses": response_list,
            "total_papers": len(response_list)
        }

