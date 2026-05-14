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
from query.question_embedding import embedding_user_message, get_keyword4llm


def query_node_first(global_config: GlobalConfig) -> dict:
    # 初始化客户端
    tmp_dir = Path("/tmp/m3llm_images")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # 确保采用关键词拆分的方式,即使没有在网页中选择
    global_config.rag.Q2K = True
    kword_list = get_keyword4llm(global_config.chat.client)
    node_results = process_per_keyword(kword_list, global_config)
    logger.debug(f"处理完毕的关键词和节点结果: {node_results}")
    relationship_results = process_all_node_relations(node_results, global_config)
    logger.debug(f"处理完毕的关键词和节点关系: {relationship_results}")



def process_per_keyword(kword_list: list, global_config: GlobalConfig) -> dict:
    """
    处理每个关键词，查询相关的实体信息，并返回全部关键词的结果。
    {kword: node}
    """
    results_dict = {}
    for kword in kword_list:
        logger.debug(f"Processing kword: {kword}")
        cypher = """
        MATCH (e:Entity{entity_id: $kword}) 
        RETURN e LIMIT 1
        """
        parameters = {"kword": kword}
        result = global_config.database.neo4j.client.neo4j_parameterized_query(cypher,parameters)
        # logger.debug(f"Cypher query result for {kword}: {result}")
        node = None
        # 提取节点内容
        if result and "node" in result and result["node"]:
            # result["node"] 是一个列表，取第一个元素的 "e"
            node_data = result["node"][0]
            node = node_data.get("e") if "e" in node_data else node_data
            logger.debug(f"Found entity for keyword {kword}: {node}")
        else:
            logger.debug(f"No entity found for keyword {kword}, use CONTAINS method")
            cypher = """
            MATCH (e:Entity)
            WHERE e.entity_id CONTAINS $kword
            RETURN e LIMIT 1
            """
            parameters = {"kword": kword}
            global_config.database.neo4j.client.update("cypher", cypher)
            result = global_config.database.neo4j.client.neo4j_parameterized_query(cypher, parameters)
            if result and "node" in result and result["node"]:
                node_data = result["node"][0]
                node = node_data.get("e") if "e" in node_data else node_data
                logger.debug(f"Found entity for keyword {kword} using CONTAINS: {node}")

        results_dict[kword] = node if node else None
    logger.debug(f"All keyword results: {results_dict}")
    return results_dict



def process_all_node_relations(node_results: dict, global_config) -> dict:
    """
    检索 node_results 中所有实体节点两两之间的无向关系。
    返回结构：{(id1, id2): path_data or None}
    """
    from itertools import combinations
    
    relation_results = {}
    nodes = [node for node in node_results.values() if node]
    logger.info(f"Total nodes to check relations: {len(nodes)}")

    # 获取所有两两组合
    for node1, node2 in combinations(nodes, 2):
        id1, id2 = node1.get("entity_id"), node2.get("entity_id")
        if not id1 or not id2:
            logger.debug(f"跳过无 entity_id 的节点对: {node1}, {node2}")
            continue

        cypher = """
        MATCH (a:Entity {entity_id: $id1}), (b:Entity {entity_id: $id2})
        MATCH path = (a)-[*1]-(b)
        RETURN path LIMIT 1
        """
        parameters = {"id1": id1, "id2": id2}

        result = global_config.database.neo4j.client.neo4j_parameterized_query(cypher, parameters)

        if result and "node" in result and result["node"]:
            path_data = result["node"][0].get("path")
            logger.debug(f"节点 {id1} 和 {id2} 之间存在关系: {path_data}")
            relation_results[(id1, id2)] = path_data
        else:
            logger.debug(f"节点 {id1} 和 {id2} 之间不存在直接关系")
            relation_results[(id1, id2)] = None

    return relation_results

