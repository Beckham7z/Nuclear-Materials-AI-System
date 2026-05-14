# -*- coding: utf-8 -*-
import asyncio
import json
from typing import List, Dict, Any
from configuration.logset import logger
from configuration.global_config import GlobalConfig
from query.question_embedding import embeddingALL,get_performance
from query.rerank_enhance import rerank_enhance
from prompt.PerformanceAnalysis import exPerformance_en, evalPerformance_en

"""Design structures with the largest negative Poisson's ratio"""

class PerformanceAnalysisHandler():
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

    async def _get_performance(self) -> Dict[str, Any]:
        """
        从给定的问题中提取性能信息，如果没有指定性能则返回空列表。
        """
        # 得到性能信息的字典
        try:
            performance_dict = await get_performance(self.global_config)
            return performance_dict
        except Exception as e:
            logger.error(f"获取性能信息失败: {str(e)}")
            return {"type": None, "performance": []}
        

    async def _get_performance_embedding(self,entity_uid_list) -> dict:
        # 将追求的性能信息进行向量化。
        try:
            original_embed = await self.embedding_client.get_response("embedding", input_list=entity_uid_list)
            return original_embed 
        except Exception as e:
            logger.error(f"调用 embedding_user_message 失败: {str(e)}")
            return {}
        
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
            AND n.entity_type CONTAINS 'mechanical properties'
            RETURN n
"""
            return self.neo4j_client.neo4j_parameterized_query(query, {"entity_uid_list": entity_uid_list})["node"]
        except Exception as e:
            logger.error(f"Neo4j 查询失败: {str(e)}")
            return []

    # MongoDB
    def _get_source_content(self, doc_id_list: List):
        if isinstance(doc_id_list, set):
            doc_id_list = list(doc_id_list)
        try:
            doc = self.mongo_client.mongo_ids_query(doc_id_list)
            return doc
        except Exception as e:
            logger.error(f"Mongo检索失败: _id={doc_id_list}, 错误: {str(e)}")
            return None
    
    def _get_mongo_full_content(self, doc_id_list: List):
        try:
            self.mongo_client.update("collection_name", "full_docts")
            results = self.mongo_client.mongo_ids_query(doc_id_list)
            logger.debug(f"MongoDB检索成功！")
            return results
        except Exception as e:
            logger.error(f"Mongo检索失败: _id={doc_id_list}, 错误: {str(e)}")
            return None
        

    async def _rerank_results(self, results:list[str]):
        """
        对查询结果进行重排序，基于相似度得分。
        """
        # 按照相似度得分降序排序
        sorted_list = await rerank_enhance(self.global_config , results)

        return sorted_list # 每个元素的评分
    
    async def _choose_best_performance(self, performance_list: List[str], doc: dict) -> dict:
        """从文本中提取指定性能指标的数值和单位"""
        try:
            # 构建分析提示
            content = doc.get('content', '')
            file_path = doc.get('file_path', '')
            
            # 检查输入有效性
            if not performance_list or not content:
                logger.warning(f"无效输入: performance_list={performance_list}, content长度={len(content)}")
                return {
                    "file_path": file_path,
                    "extracted_values": {},
                    "error": "无效输入数据"
                }
            
            # 使用性能提取提示模板
            analysis_prompt = evalPerformance_en.replace("{{performance_list}}", str(performance_list))
            analysis_prompt = analysis_prompt.replace("{{text}}", content)
            
            logger.debug(f"正在分析文档: {file_path}, 性能指标: {performance_list}")
            
            # 调用LLM进行性能数值提取
            response = await self.chat_client.get_response(
                "chat", 
                user_message=analysis_prompt
            )
            
            # 尝试解析JSON响应
            try:
                if isinstance(response, str):
                    # 清理响应中可能存在的markdown格式
                    cleaned_response = response.strip()
                    if cleaned_response.startswith("```json"):
                        cleaned_response = cleaned_response[7:]  # 移除```json
                    if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response[:-3]  # 移除```
                    
                    extracted_values = json.loads(cleaned_response)
                else:
                    extracted_values = response
                
                # 验证提取的数据格式
                if not isinstance(extracted_values, dict):
                    logger.error(f"LLM返回格式错误，期望dict，得到: {type(extracted_values)}")
                    extracted_values = {}
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {str(e)}, 原始响应: {response}")
                extracted_values = {}
            except Exception as e:
                logger.error(f"响应处理失败: {str(e)}")
                extracted_values = {}
            
            # 构建标准化返回结果
            result = {
                "file_path": file_path,
                "extracted_values": extracted_values,
                "source_content": content[:200] + "..." if len(content) > 200 else content,
                "performance_metrics_requested": performance_list,
                "extraction_success": len(extracted_values) > 0
            }
            
            logger.debug(f"文档 {file_path} 提取完成，提取到 {len(extracted_values)} 个性能数值")
            return result
        
        except Exception as e:
            logger.error(f"分析文档失败 {doc.get('file_path', 'unknown')}: {str(e)}")
            return {
                "file_path": doc.get('file_path', 'unknown'),
                "extracted_values": {},
                "error": str(e),
                "performance_metrics_requested": performance_list,
                "extraction_success": False
            }

    async def _process_documents_in_batches(self, documents: List[dict], batch_size: int, performance_list: List[str]) -> List[dict]:
        """
        分批次并发处理文档，每批最多处理 batch_size 个文档
        """
        all_results = []
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.debug(f"处理第 {batch_num}/{total_batches} 批文档，共 {len(batch)} 个")
            
            # 并发处理当前批次，传入performance_list
            tasks = [self._choose_best_performance(performance_list, doc) for doc in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理批次结果
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"批次 {batch_num} 中文档 {j} 处理失败: {str(result)}")
                    # 添加错误结果
                    all_results.append({
                        "file_path": batch[j].get('file_path', 'unknown'),
                        "extracted_values": {},
                        "error": str(result),
                        "extraction_success": False
                    })
                else:
                    all_results.append(result)
            
            logger.debug(f"批次 {batch_num} 处理完成")
            
            # 批次间短暂延迟，避免API限流
            if i + batch_size < len(documents):
                await asyncio.sleep(0.1)
        
        return all_results

    def _has_valid_performance_values(self, extracted_values: dict) -> bool:
        """
        检查extracted_values字典中是否有任意值不为"None"
        """
        if not isinstance(extracted_values, dict):
            return False
        
        # 检查是否有任意值不为"None"
        for value in extracted_values.values():
            if value != "None":
                return True
        return False
    
    async def _find_best_performance_values(self, valid_performance_results: List[Dict], performance_list: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance data to find the strongest/best performance for each metric
        
        Args:
            valid_performance_results: List containing file_path and extracted_values
            performance_list: List of performance metrics
        
        Returns:
            Dictionary with key as performance metric, value as dict containing file_path and performance value
        """
        # Initialize result dictionary
        result = {}
        for metric in performance_list:
            result[metric] = {"file_path": None, "performance_value": None}
        
        # Group data by performance metrics
        grouped_data = {}
        for metric in performance_list:
            grouped_data[metric] = []
            
        # Group data by performance metrics
        for item in valid_performance_results:
            extracted_values = item.get("extracted_values", {})
            file_path = item.get("file_path")
            
            for key, value in extracted_values.items():
                # Try to match performance metrics
                for metric in performance_list:
                    if metric.lower() in key.lower():
                        grouped_data[metric].append({
                            "file_path": file_path,
                            "key": key,
                            "value": value,
                            "extracted_values": extracted_values
                        })
                        break
        
        # Find the best value for each performance metric
        for metric, data_list in grouped_data.items():
            if not data_list:
                continue
                
            # Build LLM analysis prompt
            data_summary = []
            for i, item in enumerate(data_list):
                data_summary.append(f"{i+1}. File: {item['file_path']}, Metric: {item['key']}, Value: {item['value']}")
            
            if len(data_summary) == 1:
                # Only one result, return directly
                best_item = data_list[0]
                result[metric] = {
                    "file_path": best_item["file_path"],
                    "performance_value": best_item["value"]
                }
                continue
            
            prompt = f"""
    Analyze the following {metric} performance data and identify the one with the strongest/optimal numerical performance:

    {chr(10).join(data_summary)}

    Important guidelines:
    1. For negative Poisson's ratio, larger absolute values usually indicate stronger performance
    2. For range values (like -0.8|-0.11), consider the optimal part of the range
    3. For percentages, judge by percentage magnitude
    4. Ignore unit differences, focus on the numerical value itself
    5. Consider the physical meaning of the metric to determine what constitutes "better"

    Please respond with only the number of the optimal result (e.g., 1, 2, 3, etc.), no explanation needed.
            """
            
            try:
                # Use LLM analysis with class chat_client
                response = await self.chat_client.get_response(
                    "chat", 
                    user_message=prompt
                )
                
                # Parse LLM response, extract sequence number
                response_text = response.strip()
                
                # Try to extract numbers
                import re
                numbers = re.findall(r'\d+', response_text)
                
                if numbers:
                    best_index = int(numbers[0]) - 1  # Convert to 0-based index
                    if 0 <= best_index < len(data_list):
                        best_item = data_list[best_index]
                        result[metric] = {
                            "file_path": best_item["file_path"],
                            "performance_value": best_item["value"]
                        }
                        logger.debug(f"Found best performance for {metric}: {best_item['file_path']} - {best_item['value']}")
                    else:
                        logger.warning(f"LLM returned index {best_index} out of range, using first result")
                        best_item = data_list[0]
                        result[metric] = {
                            "file_path": best_item["file_path"],
                            "performance_value": best_item["value"]
                        }
                else:
                    logger.warning(f"Cannot parse LLM response, using first result: {response_text}")
                    best_item = data_list[0]
                    result[metric] = {
                        "file_path": best_item["file_path"],
                        "performance_value": best_item["value"]
                    }
                    
            except Exception as e:
                logger.error(f"Error analyzing {metric} performance: {str(e)}")
                # Use first result as backup when error occurs
                if data_list:
                    best_item = data_list[0]
                    result[metric] = {
                        "file_path": best_item["file_path"],
                        "performance_value": best_item["value"]
                    }
            
            # Add delay to control concurrency (concurrency number = 1)
            await asyncio.sleep(0.5)
        
        return result

    async def _analyze_best_performance(self, valid_performance_results: List[Dict], performance_list: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Wrapper function for analyzing best performance values
        """
        logger.info(f"Starting analysis of best values for {len(performance_list)} performance metrics")
        logger.info(f"Valid data count: {len(valid_performance_results)}")
        
        result = await self._find_best_performance_values(valid_performance_results, performance_list)
        
        # Output statistics
        found_count = sum(1 for v in result.values() if v["file_path"] is not None)
        logger.info(f"Successfully found best values for {found_count}/{len(performance_list)} performance metrics")
        
        # Output detailed results
        for metric, best_result in result.items():
            if best_result["file_path"]:
                logger.info(f"{metric} best performance from: {best_result['file_path']}")
                logger.debug(f"  Performance value: {best_result['performance_value']}")
            else:
                logger.warning(f"{metric} no valid data found")
        
        return result

    async def query_async(self) -> dict:
        # 多并发检索并分析性能结果。
        performance_dict = await self._get_performance()
        logger.debug(f"提取的性能信息: {performance_dict}")
        # 先直接在neo4j中查询，如果查不到再向量化后去Milvus中检索
        performance_list = performance_dict.get("performance", None)
        if len(performance_list) == 0:
            logger.debug("未指定性能，直接返回空结果")
            return {"type": performance_dict.get("type", None), "performance": []}
        entity_info = self._get_entity_info(performance_list)
        if len(entity_info) == 0:
            logger.debug("未在Neo4j中找到实体信息，尝试向量化检索")
            # 向量化检索
            embedding = await self._get_performance_embedding(performance_list)
            if not embedding:
                logger.error("向量化检索失败，返回空结果")
                return {"type": performance_dict.get("type", None), "performance": []}
            similar_vectors = self._retrieve_entities(embedding)
            if not similar_vectors:
                logger.error("Milvus检索失败，返回空结果")
                return {"type": performance_dict.get("type", None), "performance": []}
            entity_uid_list = [vec["id"] for vec in similar_vectors]
            entity_info = self._get_entity_info(entity_uid_list)
        
        logger.debug(f"从Neo4j获取的实体信息: {len(entity_info)}条")

        # 从neo4j中提取文本块，在MongoDB中检索并分析提取出最优性能的设计
        collected_source_id = []# 初始化用于存储文档内容的列表
        for node_info in entity_info:
            node = node_info["n"]
            description = node.get("description", "")
            source_id = node.get("source_id", "")
            # 按照<SEP>拆分为列表
            description_list = description.split("<SEP>")
            source_id_list = source_id.split("<SEP>")
            # 对应每个source_id获取对应的段落
            collected_source_id.extend(source_id_list)
        # 将列表转换为集合，去重
        # logger.debug(f"{collected_source_id}")
        collected_source_id = set(collected_source_id)
        logger.debug(f"收集到的数目: {len(collected_source_id)}条")
        # 批量处理 source_id
        source_contents = self._get_source_content(collected_source_id)

        logger.debug(f"开始处理 {len(source_contents)} 个文档，合并相同DOI的内容")
        merged_contents = {}
        for source_content in source_contents:
            doi = source_content['file_path']
            content = source_content['content']
            if doi not in merged_contents:
                merged_contents[doi] = content
            else:
                merged_contents[doi] += "\n" + content
        
        logger.debug(f"合并后得到 {len(merged_contents)} 个DOI文档")
        logger.debug(f"DOI列表: {list(merged_contents.keys())}")
        
        # 转换为文档列表，修正字段名
        documents = [
            {
                "content": content, 
                "file_path": doi  # 修正字段名，与 _choose_best_performance 期望的一致
            } 
            for doi, content in merged_contents.items()
        ]
                
        # 获取批次大小
        batch_size = self.global_config.rag.top_k
        logger.debug(f"使用批次大小: {batch_size}，总文档数: {len(documents)}")

        # 获取性能指标列表
        entity_uid_list = performance_dict.get("performance", [])
        logger.debug(f"性能指标列表: {entity_uid_list}")

        # 分批次并发处理文档
        performance_results = await self._process_documents_in_batches(documents, batch_size, entity_uid_list)

        # 统计结果
        successful_results = [result for result in performance_results if "error" not in result]
        failed_results = [result for result in performance_results if "error" in result]
        # 将性能有具体数值的结果提取出来
        valid_performance_results = [
            {
                'file_path': result['file_path'],
                'extracted_values': result['extracted_values']
            }
            for result in successful_results 
            if any(value != "None" for value in result.get("extracted_values", {}).values())
        ]


        logger.debug(f"处理完成: 成功 {len(successful_results)} 个, 失败 {len(failed_results)} 个")
        logger.debug(f"有具体数值的结果: {len(valid_performance_results)} 个")
        logger.debug(f"有效结果示例: {valid_performance_results[:2] if valid_performance_results else '无'}")

        # for i in valid_performance_results:
        #     if "extracted_values" in i:
        #         logger.debug(f"提取到的性能数值: {i['extracted_values']}")


        # 分析最佳性能
        if valid_performance_results:
            best_performance = await self._analyze_best_performance(valid_performance_results, entity_uid_list)
        else:
            best_performance = {}

        logger.debug(f"Best performance analysis results: {best_performance}")
        fin_doi_list = [value['file_path'] for performance, value in best_performance.items()]
        logger.debug(f"Final DOI list for full content retrieval: {fin_doi_list}")
        full_content = self._get_mongo_full_content(doc_id_list=fin_doi_list)
        # 打印前200个字符
        logger.debug(f"Full content for {full_content}: {content[0][:200]}...")
        
        return {
            "type": performance_dict.get("type", None),
            "performance": valid_performance_results,
            "best_performance": best_performance,  # 添加最佳性能分析结果
            "failed_count": len(failed_results),
            "total_processed": len(performance_results)
        }
        

