import os
import re
import platform
import streamlit as st
import numpy as np
from PIL import Image
import base64
import tempfile
from pathlib import Path

from configuration.logset import logger
from configuration.global_config import GlobalConfig
from query.question_embedding import embedding_user_message


def query_abstract_first(global_config: GlobalConfig) -> list:
    """
    检索并处理摘要相关内容，返回结果列表results_sq。
    results_sq的数据结构如下：
    [
        {
            "doi": str,                        # 当前DOI
            "score": float,                    # 检索得分
            "abstract_summary": str,           # LLM对摘要的总结
            "final_answer": str,               # LLM对所有信息的最终回答
            "best_figure": {                   # 可选，若有相关图片
                "description": str,
                "similarity_score": float,
                "base64": str,
                "tmp_path": str                # 图片本地路径
            },
            "entity_source_summaries": {       # 可选，实体source_id到LLM总结的映射
                source_id: summary
            }
        },
        ...
    ]
    """
    tmp_dir = Path("/tmp/m3llm_images")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 向量化问题
    embeddings = embedding_user_message(global_config, global_config.chat.client, global_config.embedding.client)

    milvus_client = global_config.database.milvus.client
    milvus_client.update("collection_name", "abstract")
    milvus_client.update("data", embeddings)
    similar_vectors = milvus_client.milvus_query()[0]  # 这是一个列表

    neo4j_client = global_config.database.neo4j.client
    results_sq = []

    for i, item in enumerate(similar_vectors):
        node_dict = neo4j_client.abstract_first_neo4j(item["id"])  # id为DOI
        node_dict["score"] = item.score
        node_dict["doi"] = item["id"]

        # 处理摘要、实体、图片
        abs_result = abs_first_per_doi(node_dict, global_config)

        # 构建结果字典
        result_item = {
            "doi": item["id"],
            "score": item.score,
            "abstract_summary": abs_result.get("abstract_summary", ""),
            "final_answer": abs_result.get("final_answer", "")
        }

        # 处理图片（如有）
        best_figure = abs_result.get("best_figure")
        if best_figure and best_figure.get("base64"):
            tmp_path = Path(tmp_dir) / f"similar_{i}.png"
            try:
                img_data = base64.b64decode(best_figure["base64"])
                with open(tmp_path, "wb") as f:
                    f.write(img_data)
                result_item["best_figure"] = {
                    "description": best_figure.get("description", ""),
                    "similarity_score": best_figure.get("similarity_score", 0),
                    "base64": best_figure.get("base64", ""),
                    "tmp_path": str(tmp_path)
                }
            except Exception as e:
                logger.error(f"图片解码/保存失败: {str(e)}")
        # 实体source_id到LLM总结的映射
        if "source_summary_dict" in abs_result:
            result_item["entity_source_summaries"] = abs_result["source_summary_dict"]

        results_sq.append(result_item)

    return results_sq


def abs_first_per_doi(node_dict: dict, global_config: GlobalConfig):
    """
    处理一个DOI节点及其子节点的函数，目的是逐个分析数据结构如下的字典：
    {"abstract": abstract_str,"entities": entities_list,"figures": figures_list}
    """
    results = {}

    # 1. 问题和摘要输入到LLM，得到总结
    question = global_config.rag.Question
    abstract = node_dict.get("abstract", "")
    prompt = f"Question: {question}\nAbstract: {abstract}\n请根据摘要回答问题，并简要总结。"
    global_config.chat.client.update_llmconfig("user_message", prompt)
    summary = global_config.chat.client.get_response()
    results["abstract_summary"] = summary

    # 2. 对图片描述进行相似度检索，取最高的图片
    figures = node_dict.get("figures", [])
    best_fig = None
    best_fig_score = -1
    best_fig_desc = ""
    best_fig_base64 = ""
    if figures:
        question_emb = embedding_user_message(global_config, global_config.chat.client, global_config.embedding.client)
        for fig in figures:
            desc = fig.get("description", "")
            if not desc:
                continue
            fig_emb = embedding_user_message(global_config, global_config.chat.client, global_config.embedding.client, text=desc)
            score = cosine_similarity(question_emb, fig_emb)
            if score > best_fig_score:
                best_fig_score = score
                best_fig = fig
                best_fig_desc = desc
                best_fig_base64 = fig.get("base64", "")
        if best_fig:
            results["best_figure"] = {
                "description": best_fig_desc,
                "similarity_score": best_fig_score,
                "base64": best_fig_base64
            }

    # 3. 对entity列表中的实体进行处理，提取entity_id，判断是否在问题中出现
    entities = node_dict.get("entities", [])
    matched_entities = []
    for entity in entities:
        entity_id = entity.get("entity_id", "")
        if entity_id and entity_id.lower() in question.lower():
            matched_entities.append(entity)

    # 处理出现的实体的description和source_id
    entity_desc_dict = {}
    source_text_dict = {}
    source_summary_dict = {}

    for entity in matched_entities:
        desc = entity.get("description", "")
        source_id = entity.get("source_id", "")
        desc_list = [d for d in desc.split("<SEP>") if d.strip()]
        source_id_list = [s for s in source_id.split("<SEP>") if s.strip()]
        if len(desc_list) == len(source_id_list) and len(desc_list) > 0:
            pair_dict = dict(zip(source_id_list, desc_list))
            entity_desc_dict.update(pair_dict)
            for sid in source_id_list:
                text = global_config.database.mongo.client.mongo_query("full_docts", sid)
                if text:
                    source_text_dict[sid] = text
                    single_message = (
                        f"Question: {question}\n"
                        f"Entity Description: {entity_desc_dict[sid]}\n"
                        f"Source Text: {text}\n"
                        "请结合实体描述和源文本，针对问题进行总结，只允许描述问题中出现的内容，如果不存在相关内容，则输出”问题与源文本不相关“。"
                    )
                    global_config.chat.client.update_llmconfig("user_message", single_message)
                    summary = global_config.chat.client.get_response()
                    source_summary_dict[sid] = summary

    # 合并所有信息为message
    message = f"Question: {question}\n"
    message += f"Abstract Summary: {summary}\n"
    if "best_figure" in results:
        message += f"Best Figure Description: {results['best_figure']['description']}\n"
    if entity_desc_dict:
        message += "Entity Source Descriptions:\n"
        for sid, desc in entity_desc_dict.items():
            message += f"  - Source ID: {sid}, Description: {desc}\n"
    if source_summary_dict:
        message += "Source Summaries:\n"
        for sid, summ in source_summary_dict.items():
            message += f"  - Source ID: {sid}, Summary: {summ}\n"

    # 用LLM生成最终回答
    global_config.chat.client.update_llmconfig("user_message", message)
    response = global_config.chat.client.get_response()
    results["final_answer"] = response
    results["source_summary_dict"] = source_summary_dict  # 便于主函数引用


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)