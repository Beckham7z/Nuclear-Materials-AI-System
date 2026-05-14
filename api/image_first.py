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
from llm.async_utils import run_async
import asyncio

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

    async def _get_query_embedding(self) -> Any:
        try:
            return await embedding_user_message(self.global_config, self.chat_client, self.embedding_client)
        except Exception as e:
            logger.error(f"调用 embedding_user_message 失败: {str(e)}")
            return None
    
    def _retrieve_entities(self, embeddings: Any, ) -> List[Dict[str, Any]]:
        try:
            self.milvus_client.update("top_k",self.global_config.rag.top_k)
            self.milvus_client.update("collection_name", "figure")
            self.milvus_client.update("data", embeddings)
            milvus_result = self.milvus_client.milvus_query()
            similar_vectors = [item for sublist in milvus_result for item in sublist]
            logger.debug(f"检索到的相似向量{similar_vectors}")
            return similar_vectors
        except Exception as e:
            logger.error(f"Milvus实体检索失败: {str(e)}")
            return []
    
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
    
    async def _rerank_results(self, results:list[str]):
        # 按照相似度得分降序排序
        return await rerank_enhance(self.global_config , results)

    async def _per_image_summary(self, image_info) -> dict:
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
        try:
            summary = await self.chat_client.get_response("chat", user_message=user_message)
            return {
                "tmp_path": str(tmp_path),
                "summary": summary,
                "rerank_score": image_info.get("rerank_score", 0),
                "similarity_score": image_info.get("distance", 0),
                "description": image_info.get("description", ""),
                "doi": image_info.get("doi", "")
            }
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            return {}

    async def _get_final_response(self, results: List[Dict[str, Any]]) -> str:
        final_message = f"""You are an academic assistant. Please carefully analyze the following image summaries and synthesize a comprehensive academic response.

        Question: {self.global_config.rag.Question}

        Based on the following image summaries:
        {chr(10).join(f'Image {i+1}:/n{r["summary"]}' for i, r in enumerate(results))}

        Your response should be structured as follows:
        1. **Introduction**: Briefly introduce the overall context and significance of the question.
        2. **Detailed Analysis of Each Image**: For each image, clearly and thoroughly summarize the key points, findings, and their academic implications. Please use academic language and ensure each image's contribution is explicitly discussed.
        3. **Conclusion**: Provide an integrated summary, highlighting the main insights and their relevance to the question.

        Please ensure your answer is rigorous, objective, and meets academic standards.
        """
        try:
            summary = await self.chat_client.get_response("chat", user_message=final_message, kwargs={"temperature":0.7, "max_tokens": 4096,"stream":False})
            return summary
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            return ""

    async def _ImageFirstQuery(self):
        embeddings = await self._get_query_embedding()
        similar_vectors = self._retrieve_entities(embeddings)
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
        descriptions = [item['n'].get('description', '') for item in image_node_info]
        scores = await self._rerank_results(descriptions)
        scored_items = list(zip(image_node_info, scores, distances))
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
        # 并发处理图片摘要
        results = await asyncio.gather(*[self._per_image_summary(node) for node in top_k_sorted_info])
        final_response = await self._get_final_response(results)
        results.append(final_response)
        return results

    def Page_Display(self):
        # 检查 run_async 返回值，如果是协程对象则用 asyncio 运行
        result_or_coro = run_async(self._ImageFirstQuery())
        results = None
        if asyncio.iscoroutine(result_or_coro):
            import nest_asyncio
            nest_asyncio.apply()
            try:
                loop = asyncio.get_event_loop()
                # 如果loop已关闭则新建
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                results = loop.run_until_complete(result_or_coro)
            except RuntimeError:
                # 若再次遇到事件循环问题，强制新建事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(result_or_coro)
        else:
            results = result_or_coro
        try:
            if results:
                st.write("### Retrieved Images and Their Summaries:")
                for i, result in enumerate(results[:-1]):
                    with st.container():
                        with st.expander(f"Image {i+1} (Score: {result['similarity_score']:.4f}) (Rerank: {result['rerank_score']:.4f})", expanded=True):
                            try:
                                image = Image.open(result['tmp_path'])
                                st.image(
                                    image,
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"无法加载图片 {result['tmp_path']}: {str(e)}")
                                continue
                            if result.get('doi'):
                                st.markdown(f"**DOI:** {result['doi']}")
                            st.markdown("**Image Summary:**")
                            st.markdown(result['summary'])
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                return {
                    "text": "未找到相关结果",
                    "images": []
                }
            st.markdown("### Comprehensive Analysis")
            st.markdown("---")
            st.markdown(results[-1])
        except Exception as e:
            st.error(f"生成最终回答失败: {str(e)}")
            return {
                "text": "生成回答时发生错误",
                "results": []
            }

