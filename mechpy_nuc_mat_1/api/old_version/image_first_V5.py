import re
import uuid
import base64
from PIL import Image
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any
from configuration.logset import logger
from configuration.global_config import GlobalConfig
from query.question_embedding import embedding_user_message
from query.rerank_enhance import rerank_enhance

class ImageQueryHandler:
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self.tmp_dir = Path("/tmp/m3llm_images")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.milvus_client = self.global_config.database.milvus.client
        self.neo4j_client = self.global_config.database.neo4j.client
        self.chat_client = self.global_config.chat.client
        self.embedding_client = self.global_config.embedding.client
        self.rerank_client = self.global_config.rerank.client

    # 问题向量化
    def _get_query_embedding(self) -> Any:
        try:
            return embedding_user_message(self.global_config, self.chat_client, self.embedding_client)
        except Exception as e:
            logger.error(f"调用 embedding_user_message 失败: {str(e)}")
            return None
        
    # Milvus检索
    def _retrieve_entities(self, embeddings: Any, ) -> List[Dict[str, Any]]:
        try:
            self.milvus_client.update("top_k",self.global_config.rag.top_k)
            self.milvus_client.update("collection_name", "figure")
            self.milvus_client.update("data", embeddings)
            
            milvus_result = self.milvus_client.milvus_query()
            # 将检索到的所有向量保存到列表中
            similar_vectors = [item for sublist in milvus_result for item in sublist]
            logger.debug(f"检索到的相似向量{similar_vectors}")
            return similar_vectors
        except Exception as e:
            logger.error(f"Milvus实体检索失败: {str(e)}")
            return []
        
    # Neo4j检索
    def _retrieve_image_info(self, doi_list: List[str],path_list: List[str]) -> List[Dict[str, Any]]:
        query = """
            MATCH (n:Figure)
            WHERE n.doi IN $doi_list AND n.image_path IN $path_list
            RETURN n
        """
        try:
            return self.neo4j_client.neo4j_parameterized_query(query, {"doi_list": doi_list,"path_list":path_list})["node"]
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
    
    def _per_image_summary(self,image_info) -> str:
        """输入是单个Figure节点的信息，返回处理好的单个节点输出的字典"""
        tmp_path = self.tmp_dir / f"{uuid.uuid4().hex[:8]}.png"
        try:
            img_data = base64.b64decode(image_info["image_base64"])
            with open(tmp_path, "wb") as f:
                f.write(img_data)
        except Exception as e:
            logger.error(f"图片解码/保存失败: {str(e)}")
            return {}
        
        user_message = f"""
            Question: {self.global_config.rag.Question}
            Image Description: {image_info.get('description', '')}
            Paper DOI: {image_info.get('doi', '')}
            Please provide a brief summary (within 100 words) focusing on how this image relates to the question.
            """
        self.global_config.chat.client.update_llmconfig("user_message", user_message)
        try:
            response = self.chat_client.get_response()
            return {
                    "tmp_path": str(tmp_path),
                    "summary": response,  # 改为summary
                    "rerank_score": image_info.get("rerank_score", 0),
                    "similarity_score": image_info.get("distance", 0),
                    "description": image_info.get("description", ""),
                    "doi": image_info.get("doi", "")
                }
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")

    def _get_final_response(self, results: List[Dict[str, Any]]) -> str:
        final_message = f"""You are an academic assistant. Please carefully analyze the following image summaries and synthesize a comprehensive academic response.

        Question: {self.global_config.rag.Question}

        Based on the following image summaries:
        {chr(10).join(f'Image {i+1}:\n{r["summary"]}' for i, r in enumerate(results))}

        Your response should be structured as follows:
        1. **Introduction**: Briefly introduce the overall context and significance of the question.
        2. **Detailed Analysis of Each Image**: For each image, clearly and thoroughly summarize the key points, findings, and their academic implications. Please use academic language and ensure each image's contribution is explicitly discussed.
        3. **Conclusion**: Provide an integrated summary, highlighting the main insights and their relevance to the question.

        Please ensure your answer is rigorous, objective, and meets academic standards.
        """
        self.chat_client.update_llmconfig("user_message", final_message)
        self.chat_client.update_llmconfig("kwargs", {"temperature":0.7, "max_tokens": 4096,"stream":False})
        final_response = self.chat_client.get_response()
        return final_response
        
    def _ImageFirstQuery(self):
        embeddings = self._get_query_embedding()
        similar_vectors = self._retrieve_entities(embeddings)
        # 从 Milvus结果 中提取对应的 distance
        distances = [item.get('distance', None) for item in similar_vectors]
        doi_list = [re.sub(r"/paper.*$", "", item.get("id", "")) for item in similar_vectors]
        path_list = [
            re.search(r"/paper/auto/(.+)", item.get("id", "")).group(1)
            for item in similar_vectors
            if re.search(r"/paper/auto/(.+)", item.get("id", ""))
        ]
        image_node_info = self._retrieve_image_info(doi_list, path_list)
        for item in image_node_info:
            logger.debug(f"检索到的节点 {item['n']['doi']} {item['n']['image_path']}")


        # 对图像节点信息进行重排序
        descriptions = [item['n'].get('description', '') for item in image_node_info]
        scores = self._rerank_results(descriptions)

        scored_items = list(zip(image_node_info, scores, distances))
        # 对 scored_items 按照得分进行排序
        sorted_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
        sorted_info = [
                        {
                            'doi': item['n'].get('doi', ''),
                            'image_path': item['n'].get('image_path', ''),
                            'description': item['n'].get('description', ''),
                            "image_base64": item['n'].get('base64', ''),
                            'distance': distance,
                            'rerank_score': score
                        }
                        for item, score, distance in sorted_items
                    ]

        top_k_sorted_info = sorted_info[:self.global_config.rag.top_k]
        results = []
        for node in top_k_sorted_info:
            # 打印调试信息
            logger.debug(f"重排序后节点：DOI={node['doi']}, Score={node['rerank_score']:.4f}, Distance={node['distance']}")
            # logger.debug(f"image_base64 ={node['image_base64']} ")
            # 打印node 的详细信息
            logger.debug(f"Node details: {node.keys()}")
            result = self._per_image_summary(node)
            results.append(result)
        
        # 合并汇总返回总结
        final_response = self._get_final_response(results)
        results.append(final_response)

        return results
    
    def Page_Display(self):
        results = self._ImageFirstQuery()
        try:
            # 首先显示图片和对应的总结
            if results:
                st.write("### Retrieved Images and Their Summaries:")
                for i, result in enumerate(results[:-1]):
                    with st.container():
                        # 使用expander来组织内容
                        with st.expander(f"Image {i+1} (Score: {result['similarity_score']:.4f}) (Rerank: {result['rerank_score']:.4f})", expanded=True):
                            # 显示图片
                            try:
                                image = Image.open(result['tmp_path'])
                                st.image(
                                    image,
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"无法加载图片 {result['tmp_path']}: {str(e)}")
                                continue
                            
                            # 显示图片相关信息
                            if result.get('doi'):
                                st.markdown(f"**DOI:** {result['doi']}")
                            st.markdown("**Image Summary:**")
                            st.markdown(result['summary'])
                        
                        # 添加间距
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                return {
                    "text": "未找到相关结果",
                    "images": []
                }
            st.markdown("### Comprehensive Analysis")
            st.markdown("---")
            st.markdown(results[-1])  # 最后一个元素是综合分析的回答
            
        except Exception as e:
            st.error(f"生成最终回答失败: {str(e)}")
            return {
                "text": "生成回答时发生错误",
                "results": []
            }
            
