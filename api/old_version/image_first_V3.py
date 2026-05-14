import os
import re
import platform
import streamlit as st
import numpy as np
from PIL import Image
import base64
import tempfile
from pathlib import Path
from pymilvus import MilvusClient
from neo4j import GraphDatabase
from pymongo import MongoClient

from typing import Tuple, Optional
from schemas.config import LLMConfig , DatabaseConfig, Neo4jConfig, MongoConfig, MilvusConfig, GlobalConfig
from llm.chat_ollama import embedding_llm_complete_ollama
from llm.chat_llm import initialize_chat_llm



def process_image_from_response(global_config: GlobalConfig) -> dict:
    """主处理函数"""
    # 1. 初始化客户端
    client_dict = create_client_from_config(global_config)
    
    # 2. 问题向量化
    embeddings = embedding_user_message(global_config.embedding, client_dict["embed_client"])
    if embeddings is None:
        return {"error": "向量化失败"}
    
    # 3. Milvus检索
    similar_vectors = search_similar_vectors(
        embeddings, 
        client_dict["milvus_client"]
    )
    
    # 创建固定的临时目录
    tmp_dir = Path("/tmp/m3llm_images")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    # 4. 处理每个相似结果
    for i, item in enumerate(similar_vectors):
        # 5. 从Neo4j获取图片信息
        image_info = get_image_info_from_neo4j(
            item.id, 
            client_dict["neo4j_client"]
        )
        image_info["score"] = item.score
        
        # 6. 处理单张图片
        result = process_single_image(
            image_info,
            global_config.chat.user_message,
            global_config.chat,
            client_dict["chat_client"],
            str(tmp_dir),
            i
        )
        if result:
            results.append(result)
    
    # 7. 生成最终回答
    final_result = generate_final_response(
        results,
        global_config.chat.user_message,
        global_config.chat,
        client_dict["chat_client"]
    )
    
    # 8. 清理旧的临时文件，但保留当前会话的文件
    current_files = {Path(result.get('tmp_path')).name for result in results if result.get('tmp_path')}
    for file_path in tmp_dir.glob("similar_*.png"):
        if file_path.name not in current_files:
            try:
                file_path.unlink()
            except Exception:
                pass
    
    return final_result




def create_client_from_config(global_config: GlobalConfig) -> dict:
    """
    根据全局配置创建所有需要的客户端
    """
    clients = {}
    
    # 初始化LLM客户端
    try:
        clients["chat_client"] = initialize_chat_llm(global_config.chat)
    except Exception as e:
        st.error(f"Chat客户端初始化失败: {str(e)}")
        clients["chat_client"] = None

    try:
        clients["embed_client"] = initialize_chat_llm(global_config.embedding)
    except Exception as e:
        st.error(f"Embedding客户端初始化失败: {str(e)}")
        clients["embed_client"] = None

    try:
        clients["vl_client"] = initialize_chat_llm(global_config.vl)
    except Exception as e:
        st.error(f"VL客户端初始化失败: {str(e)}")
        clients["vl_client"] = None

    try:
        clients["rerank_client"] = initialize_chat_llm(global_config.rerank)
    except Exception as e:
        st.error(f"Rerank客户端初始化失败: {str(e)}")
        clients["rerank_client"] = None

    # 初始化Milvus客户端
    try:
        milvus_conf = global_config.database.milvus
        milvus_token = f"{milvus_conf.username}:{milvus_conf.password}"
        clients["milvus_client"] = MilvusClient(
            uri=milvus_conf.url,
            token=milvus_token
        )
        clients["milvus_client"].use_database(db_name=milvus_conf.database)
        clients["milvus_client"].load_collection("fig_collection")
    except Exception as e:
        st.error(f"Milvus客户端初始化失败: {str(e)}")
        clients["milvus_client"] = None

    # 初始化Neo4j客户端
    try:
        neo4j_conf = global_config.database.neo4j
        clients["neo4j_client"] = GraphDatabase.driver(
            neo4j_conf.url,
            auth=(neo4j_conf.username, neo4j_conf.password)
        )
    except Exception as e:
        st.error(f"Neo4j客户端初始化失败: {str(e)}")
        clients["neo4j_client"] = None

    return clients

def embedding_user_message(embed_config: LLMConfig, embed_client=None) -> Optional[list]:
    """
    将用户消息向量化
    
    Args:
        embed_config: Embedding模型配置
        embed_client: Embedding客户端
        
    Returns:
        Optional[list]: 向量化结果，失败则返回None
    """
    try:
        if embed_config.institution.lower() == "ollama":
            embeddings_list = embedding_llm_complete_ollama(embed_config, embed_client)
            # 检查向量是否有效
            if isinstance(embeddings_list, (list, np.ndarray)) and len(embeddings_list) > 0:
                return embeddings_list.tolist() if isinstance(embeddings_list, np.ndarray) else embeddings_list
        else:
            st.warning(f"不支持的Embedding机构: {embed_config.institution}")
            
        return None
        
    except Exception as e:
        st.error(f"向量化失败: {str(e)}")
        return None

def search_similar_vectors(embeddings: list, milvus_client: MilvusClient, top_k: int = 5) -> list:
    """在Milvus中搜索相似向量"""
    try:
        search_results = milvus_client.search(
            collection_name="fig_collection",
            data=[embeddings],
            limit=top_k,
            search_params={
                "metric_type": "COSINE", 
                "params": {}
            },
            output_fields=["fig_id", "fig_varchar"]
        )
        return search_results[0]  # 返回第一个查询的结果
    except Exception as e:
        st.error(f"Milvus搜索失败: {str(e)}")
        return []

def get_image_info_from_neo4j(fig_id: int, neo4j_client: GraphDatabase.driver) -> dict:
    """从Neo4j获取图片信息,包括关联的DOI信息"""
    def query_node_by_id(tx, node_id):
        # 首先打印节点的所有属性，用于调试
        debug_query = """
        MATCH (n:Figure)
        WHERE id(n) = $node_id
        RETURN properties(n) as props
        """
        debug_result = tx.run(debug_query, node_id=node_id).data()
        if debug_result:
            print("节点属性:", debug_result[0]['props'])

        # 修改后的查询语句，获取所有需要的信息
        query = """
        MATCH (n:Figure)
        WHERE id(n) = $node_id
        OPTIONAL MATCH (n)<-[r]-(d:DOI)
        RETURN n, 
               n.image as image_base64,  // 尝试其他可能的属性名
               n.image_data as backup_image,
               n.description as description,
               d.id as doi_id
        """
        result = tx.run(query, node_id=node_id)
        return result.data()

    try:
        with neo4j_client.session() as session:
            result = session.execute_read(query_node_by_id, fig_id)
            if result:
                node_data = result[0]
                # 尝试不同的属性名来获取图片数据
                image_base64 = (
                    node_data.get('image_base64') or 
                    node_data.get('backup_image') or 
                    node_data['n'].get('image') or 
                    node_data['n'].get('image_data') or 
                    node_data['n'].get('image_base64')
                )
                
                if not image_base64:
                    st.warning(f"找到节点但没有图片数据, 节点ID: {fig_id}")
                    print("可用的属性:", node_data['n'].keys())
                    return {}

                return {
                    "image_base64": image_base64,
                    "description": node_data.get('description') or node_data['n'].get('description', ''),
                    "doi": node_data.get('doi_id', '')
                }
            
            st.warning(f"未找到节点, ID: {fig_id}")
            return {}
            
    except Exception as e:
        st.error(f"Neo4j查询失败: {str(e)}")
        print(f"详细错误: {str(e)}")
        return {}

def process_single_image(image_info: dict, question: str, chat_config, chat_client, tmp_dir: str, index: int) -> dict:
    """处理单张图片并生成回答"""
    try:
        # 检查必要字段
        if not image_info.get("image_base64"):
            st.error("图片数据为空")
            return {}
            
        # 保存base64图片到临时目录
        tmp_path = Path(tmp_dir) / f"similar_{index}.png"
        try:
            img_data = base64.b64decode(image_info["image_base64"])
            with open(tmp_path, "wb") as f:
                f.write(img_data)
        except Exception as e:
            st.error(f"图片解码/保存失败: {str(e)}")
            return {}
        
        # 构建针对单张图片的提示词
        prompt = f"""
        Question: {question}
        Image Description: {image_info.get('description', '')}
        Paper DOI: {image_info.get('doi', '')}
        Similarity Score: {image_info.get('score', 0)}
        
        Please provide a brief summary (within 100 words) focusing on how this image relates to the question.
        """
        
        # 获取LLM的单图总结
        try:
            response = chat_client.chat.completions.create(
                model=chat_config.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that explains scientific figures concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            return {
                "tmp_path": str(tmp_path),
                "summary": response.choices[0].message.content,  # 改为summary
                "similarity_score": image_info.get("score", 0),
                "description": image_info.get("description", ""),
                "doi": image_info.get("doi", "")
            }
        except Exception as e:
            st.error(f"LLM调用失败: {str(e)}")
            return {}
            
    except Exception as e:
        st.error(f"图片处理失败: {str(e)}")
        return {}

def generate_final_response(results: list, question: str, chat_config: LLMConfig, chat_client) -> dict:
    """生成最终的综合回答,并包含图片展示"""
    if not results:
        return {
            "text": "未找到相关结果",
            "images": []
        }
    
    try:
        # 首先显示图片和对应的总结
        if results:
            st.write("### Retrieved Images and Their Summaries:")
            for i, result in enumerate(results):
                with st.container():
                    # 使用expander来组织内容
                    with st.expander(f"Image {i+1} (Score: {result['similarity_score']:.4f})", expanded=True):
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
        
        # 然后生成整体总结
        final_prompt = f"""Question: {question}

Based on the following image summaries:

{chr(10).join(f'Image {i+1}:\n{r["summary"]}' for i, r in enumerate(results))}

Please provide a comprehensive answer synthesizing all the information above. 
Structure your response with:
1. A clear introduction
2. Key points from each image
3. A conclusion
"""
        
        # 生成总体回答
        try:
            response = chat_client.chat.completions.create(
                model=chat_config.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that synthesizes information from scientific images."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7,
                max_tokens=4096,
                stream=False  # 禁用流式响应以提高稳定性
            )
            
            # 显示总体回答
            st.markdown("### Comprehensive Analysis")
            st.markdown("---")
            full_response = response.choices[0].message.content
            st.markdown(full_response)
            
            return {
                "text": full_response,
                "results": results
            }
            
        except Exception as e:
            st.error(f"LLM响应生成失败: {str(e)}")
            return {
                "text": "生成总结失败，但您仍可以查看上方的图片分析。",
                "results": results
            }
            
    except Exception as e:
        st.error(f"生成最终回答失败: {str(e)}")
        return {
            "text": "生成回答时发生错误",
            "results": []
        }