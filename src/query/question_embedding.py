# -*- coding: utf-8 -*-
import json
import numpy as np
from typing import Tuple, Optional
from configuration.logset import logger
from configuration.global_config import GlobalConfig
from prompt.ex_keyword import EX_kword_en
from prompt.PerformanceAnalysis import exPerformance_en
import asyncio

async def embedding_user_message(globalconfig: GlobalConfig, chat_client=None, embed_client=None) -> Optional[list]:
    """
    将用户消息向量化
    
    Args:
        globalconfig: 全局配置
        embed_client: Embedding客户端
        
    Returns:
        Optional[list]: 向量化结果，失败则返回空列表
    """
    logger.debug(f"正在将问题向量化")
    if globalconfig.rag.Q2K:
        kword_list = await get_keyword4llm(chat_client)
        if len(kword_list) == 0:
            logger.debug(f"关键词提取失败: {kword_list}使用默认检索方式")
            embedlist = await embed_client.get_response("embedding", input_list=globalconfig.rag.Question)
        else:
            logger.debug(f"关键词提取成功: {kword_list}使用关键词检索方式")
            embedlist = await embed_client.get_response("embedding", input_list=kword_list)
    else:
        logger.debug(f"正在使用默认检索方式处理用户问题: {globalconfig.rag.Question}")
        embedlist = await embed_client.get_response("embedding", input_list=globalconfig.rag.Question)
    return embedlist

async def embeddingALL(globalconfig: GlobalConfig, chat_client=None, embed_client=None) -> Optional[dict]:
    """
    返回原始问题和关键词的全部向量字典
    
    Args:
        globalconfig: 全局配置
        chat_client: 聊天客户端
        embed_client: Embedding客户端
        
    Returns:
        Optional[dict]: 包含原始问题和关键词向量的字典，格式为 {
            "问题文本": [问题向量],
            "关键词1": [关键词1向量],
            "关键词2": [关键词2向量],
            ...
        }
    """
    logger.debug(f"正在获取所有向量化结果")
    result = {}
    
    # 首先获取原始问题的向量
    original_question = globalconfig.rag.Question
    if isinstance(original_question, list):
        original_question = original_question[0] if original_question else ""
    
    original_embed = await embed_client.get_response("embedding", input_list=[original_question])
    if original_embed and len(original_embed) > 0:
        result[original_question] = original_embed[0]
    
    if globalconfig.rag.Q2K:
        # 获取关键词并向量化
        kword_list = await get_keyword4llm(chat_client)
        if len(kword_list) > 0:
            logger.debug(f"关键词提取成功: {kword_list}")
            keywords_embed = await embed_client.get_response("embedding", input_list=kword_list)
            if keywords_embed and len(keywords_embed) == len(kword_list):
                for keyword, embedding in zip(kword_list, keywords_embed):
                    result[keyword] = embedding
        else:
            logger.debug(f"关键词提取失败，只返回原始问题向量")
    
    return result

async def get_keyword4llm(chat_client=None, max_retries=5):
    """
    尝试多次从 chat_client 获取关键词列表，并将字典按评分转为列表返回。
    """
    old_user_message = chat_client.llm_config.user_message
    prompt = EX_kword_en.replace("{input_question}", str(old_user_message))
    await chat_client.get_response("chat", user_message=prompt)
    for attempt in range(max_retries):
        logger.debug(f"get_keyword4llm attempt {attempt+1}: {chat_client}")
        response = await chat_client.get_response("chat", user_message=prompt)

        cleaned_response = extract_json_block(response)

        logger.debug(f"cleaned_response: {cleaned_response}")
        if cleaned_response:
            try:
                # 安全地将字符串转为字典
                keywords_dict = eval(cleaned_response)
                if isinstance(keywords_dict, dict):
                    keyword_items = keywords_dict.get("keyword", {})
                    if isinstance(keyword_items, dict):
                        # 按评分降序排序并转为列表
                        sorted_keywords = sorted(keyword_items.items(), key=lambda x: x[1], reverse=True)
                        logger.info(f"Extracted keywords: {sorted_keywords}")
                        return [k for k, v in sorted_keywords]
                    else:
                        logger.error("Extracted 'keyword' is not a dict.")
                        return []
                else:
                    logger.error("Extracted keywords are not in dict format.")
                    return []
            except Exception as e:
                logger.error(f"Error evaluating extracted keywords: {e}")
                return []
        else:
            logger.debug("No valid JSON block found in the response, retrying...")
    chat_client.llm_config.update_llmconfig("user_message",old_user_message)
    logger.error("Failed to extract valid JSON block after multiple attempts.")
    return []

async def get_performance(globalconfig: GlobalConfig, max_retries=5):
    """
    从给定的问题中提取性能信息，如果没有指定性能则返回空列表。
    最多重试5次直到获得正确格式的响应。
    """
    if not globalconfig.rag.Question: # 如果问题为空，则直接返回下述字典
        logger.debug("No question provided, returning empty performance dictionary.")
        return {"type": None, "performance": []}
    
    prompt = exPerformance_en.replace("{text}", globalconfig.rag.Question)
    
    for attempt in range(max_retries):
        logger.debug(f"get_performance attempt {attempt+1}")
        
        try:
            response = await globalconfig.chat.client.get_response("chat", user_message=prompt)
            cleaned_response = extract_json_block(response)
            logger.debug(f"cleaned_response (attempt {attempt+1}): {cleaned_response}")
            
            # 检查回复必须为下述格式：{"type": str, "performance": []}
            if cleaned_response:
                try:
                    # 使用json.loads替代eval，更安全
                    performance_dict = json.loads(cleaned_response)
                    
                    # 验证格式是否正确
                    if (isinstance(performance_dict, dict) and 
                        "type" in performance_dict and 
                        "performance" in performance_dict and
                        isinstance(performance_dict["performance"], list)):
                        
                        logger.debug(f"Successfully extracted performance on attempt {attempt+1}: {performance_dict}")
                        return performance_dict
                    else:
                        logger.warning(f"Attempt {attempt+1}: Invalid format - missing required keys or wrong types")
                        logger.debug(f"performance_dict: {performance_dict}")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempt+1}: JSON decode error: {e}")
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1}: Unexpected error: {e}")
            else:
                logger.warning(f"Attempt {attempt+1}: No valid JSON block found in the response.")
                
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: Error getting response from chat client: {e}")
        
        # 如果不是最后一次尝试，等待一小段时间再重试
        if attempt < max_retries - 1:
            logger.debug(f"Retrying in 1 second... (attempt {attempt+2}/{max_retries})")
            await asyncio.sleep(1)
    
    # 所有重试都失败后，返回默认值
    logger.error(f"Failed to extract valid performance format after {max_retries} attempts.")
    return {"type": None, "performance": []}
    

    
    



import re

def extract_json_block(text: str) -> str:
    """
    先删除<think>和</think>之间的部分（如果有），
    然后尝试提取字符串中 ```json 到 ``` 之间的内容（如果有），
    如果没有则尝试提取 {"keyword": 到} 之间的内容。
    如果都没有，返回空字符串。
    """
    import re

    # 1. 删除<think>...</think>之间的内容
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. 提取 ```json ... ``` 之间的内容
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # # 3. 提取 {"keyword": ...} 之间的内容
    # match = re.search(r'(\{"keyword":.*?\})', text, re.DOTALL)
    # if match:
    #     return match.group(1).strip()

    return ""