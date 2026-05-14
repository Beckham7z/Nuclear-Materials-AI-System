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
          
def query_image_first(global_config: GlobalConfig) -> dict:
    tmp_dir = Path("/tmp/m3llm_images")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # 向量化问题
    embeddings = embedding_user_message(global_config, global_config.chat.client, global_config.embedding.client)
    # Milvus检索

    milvus_client =  global_config.database.milvus.client
    milvus_client.update("collection_name", "figure")
    milvus_client.update("data", embeddings)
    similar_vectors = milvus_client.milvus_query()[0] # 这是一个列表
    logger.debug(f"检索到的相似向量{similar_vectors}")
    # 进行图数据库检索
    neo4j_client  =  global_config.database.neo4j.client
    # 处理检索到的向量
    results_sq = []
    for i, item in enumerate(similar_vectors):
        match = re.match(r"^(.*?)/paper/auto/(.*)$", item["id"])
        doi = match.group(1)
        path = match.group(2)
        # logger.debug(f"doi和路径{doi},{path}")
        node_data = neo4j_client.image_first_neo4j(doi,path)["image_node"][0] # 第一个
        # logger.debug(f"检索到的节点数据{node_data}")
        image_info = {
                    "image_base64": node_data['a'].get('base64', ''),
                    "description": node_data['a'].get('description', ''),
                    "doi": node_data['a'].get('doi', ''),
                    "score": item.score, # 相似度得分
                }
        # logger.error(f"处理的图片信息: {image_info}")
        tmp_path = Path(tmp_dir) / f"similar_{i}.png"
        try:
            img_data = base64.b64decode(image_info["image_base64"])
            with open(tmp_path, "wb") as f:
                f.write(img_data)
        except Exception as e:
            logger.error(f"图片解码/保存失败: {str(e)}")
            return {}
        prompt = f"""
            Question: {global_config.rag.Question}
            Image Description: {image_info.get('description', '')}
            Paper DOI: {image_info.get('doi', '')}
            Please provide a brief summary (within 100 words) focusing on how this image relates to the question.
            """
        global_config.chat.client.update_llmconfig("user_message", prompt)
        try:
            response = global_config.chat.client.get_response()
            results_sq.append( {
                    "tmp_path": str(tmp_path),
                    "summary": response,  # 改为summary
                    "similarity_score": image_info.get("score", 0),
                    "description": image_info.get("description", ""),
                    "doi": image_info.get("doi", "")
                })
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            results_sq.append({})
    return results_sq