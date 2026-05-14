#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MongoDB工具函数 - 专门用于MongoDB文档检索和查询
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
import logging

# 创建logger
logger = logging.getLogger(__name__)


def search_mongo_documents(query: str, limit: int = 10) -> list:
    """从MongoDB中搜索相关文档"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['mech']
        collection = db['md_documents']
        
        # 使用正则表达式进行文本搜索
        regex_query = {"content": {"$regex": query, "$options": "i"}}
        documents = list(collection.find(regex_query).limit(limit))
        
        # 如果没有找到结果，尝试更宽松的搜索
        if not documents:
            keywords = query.split()
            if len(keywords) > 1:
                # 尝试搜索部分关键词
                for keyword in keywords:
                    if len(keyword) > 3:  # 只搜索长度大于3的关键词
                        partial_query = {"content": {"$regex": keyword, "$options": "i"}}
                        partial_results = list(collection.find(partial_query).limit(limit))
                        documents.extend(partial_results)
                        if len(documents) >= limit:
                            break
        
        # 去重并限制数量
        seen_ids = set()
        unique_documents = []
        for doc in documents:
            if doc['_id'] not in seen_ids:
                seen_ids.add(doc['_id'])
                unique_documents.append(doc)
                if len(unique_documents) >= limit:
                    break
        
        client.close()
        
        logger.info(f"从MongoDB检索到 {len(unique_documents)} 个相关文档")
        return unique_documents
        
    except Exception as e:
        logger.error(f"MongoDB搜索失败: {e}")
        return []


def build_enhanced_prompt(user_message: str, mongo_results: list) -> str:
    # --- 新增：把 DOI 也拼进去 ---
    ref_list = "\n".join(
        f"[{i+1}] DOI: {doc.get('doi', 'N/A')}\n内容: {doc.get('content', '')[:300]}..."
        for i, doc in enumerate(mongo_results)
    )
    # -----------------------------
    
    prompt = f"""你是核电材料领域的专家，拥有深厚的材料科学和核工程知识。
请基于以下检索到的相关文档和您的专业知识，回答用户问题，并**在文末列出所有参考 DOI**。

用户问题：{user_message}

检索到的相关文档：
{ref_list}

回答要求：
1. 给出科学结论、优缺点对比、注意事项、安全标准、研究方向；
2. **文末单独一段：「参考资料 DOI」**，逐条列出；
3. 使用专业术语，保持清晰易懂。
"""
    return prompt


def handle_mongo_query(global_config):
    """处理MongoDB查询的独立功能"""
    try:
        import streamlit as st
        from llm.async_utils import run_async
        
        st.write("### MongoDB文档检索结果")
        
        # 从MongoDB检索文档
        query = global_config.rag.Question
        limit = global_config.rag.top_k
        
        # 使用log_process函数（需要在调用环境中定义）
        log_process(f"检索关键词: {query}", "MongoDB检索模块", st.empty())
        mongo_results = search_mongo_documents(query, limit)
        
        if mongo_results:
            st.success(f"✅ 成功检索到 {len(mongo_results)} 个相关文档")
            
            # 显示检索结果
            for i, doc in enumerate(mongo_results):
                with st.expander(f"文档 {i+1}: {doc.get('header', '无标题')}", expanded=False):
                    st.markdown(f"**文件路径:** {doc.get('file_path', '未知')}")
                    st.markdown(f"**内容:**")
                    st.markdown(doc.get('content', '无内容'))
                    
            # 构建增强提示词并调用AI
            enhanced_prompt = build_enhanced_prompt(query, mongo_results)
            chat_client = global_config.chat.client
            
            log_process("基于检索结果生成分析", "AI分析模块", st.empty())
            response = run_async(chat_client.get_response(
                user_message=enhanced_prompt,
                model_type="chat"
            ))
            
            st.markdown("### 基于文档检索的分析结果")
            st.markdown(response)
            
            return {
                "text": response,
                "mongo_results": mongo_results
            }
        else:
            st.warning("⚠️ 未在文档数据库中找到相关结果，将使用通用知识进行分析")
            
            # 使用基础提示词
            base_prompt = f"""你是核电材料领域的专家。请基于您的专业知识回答以下问题：

问题：{query}

请提供专业、准确的分析，并说明分析的局限性。
"""
            response = run_async(chat_client.get_response(
                user_message=base_prompt,
                model_type="chat"
            ))
            
            st.markdown("### 通用知识分析结果")
            st.markdown(response)
            
            return {
                "text": response,
                "mongo_results": []
            }
            
    except Exception as e:
        import streamlit as st
        st.error(f"MongoDB查询失败: {e}")
        return {
            "text": f"查询失败: {str(e)}",
            "mongo_results": []
        }


def log_process(message, module, placeholder, is_error=False):
    """记录处理步骤并更新显示（需要在调用环境中定义）"""
    import streamlit as st
    import uuid
    from datetime import datetime
    
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "module": module,
        "message": message,
        "is_error": is_error
    }
    
    st.session_state.process_logs.append(entry)
    
    # 更新显示
    with placeholder.container():
        for item in st.session_state.process_logs:
            if item["is_error"]:
                status = "step-error"
                icon = "❌"
            else:
                status = "step-completed"
                icon = "✅"
            
            st.markdown(f"""
            <div class="process-step {status}">
                <strong>{item['timestamp']}</strong> [{item['module']}] {icon} {item['message']}
            </div>
            """, unsafe_allow_html=True)
    
    # 模拟处理时间
    if not is_error:
        time.sleep(0.3)


def verify_mongo_data():
    """验证MongoDB中的数据"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['mech']
        collection = db['md_documents']

        # 统计总文档数
        total_count = collection.count_documents({})
        print(f'📊 数据库中的总文档数: {total_count}')

        # 显示不同文件的文档分布
        print('📁 文件分布:')
        file_paths = collection.distinct('file_path')
        for file_path in file_paths:
            count = collection.count_documents({'file_path': file_path})
            print(f'  {file_path}: {count} 个文档')

        # 显示一些示例文档
        print('📄 示例文档内容:')
        documents = list(collection.find().limit(3))
        for i, doc in enumerate(documents):
            print(f'文档 {i+1}:')
            print(f'  ID: {doc["_id"]}')
            print(f'  标题: {doc.get("header", "N/A")}')
            content_preview = doc.get('content', '')[:100] + '...' if len(doc.get('content', '')) > 100 else doc.get('content', '')
            print(f'  内容预览: {content_preview}')
            print(f'  文件路径: {doc.get("file_path", "N/A")}')
            print('  ---')

        client.close()
        return True
        
    except Exception as e:
        print(f'❌ 数据验证失败: {e}')
        return False


if __name__ == "__main__":
    # 测试功能
    print("测试MongoDB工具函数...")
    
    # 验证数据
    verify_mongo_data()
    
    # 测试搜索
    query = "CLAM steel irradiation"
    results = search_mongo_documents(query, limit=3)
    print(f"搜索 '{query}' 找到 {len(results)} 个文档")
    
    # 测试提示词构建
    prompt = build_enhanced_prompt(query, results)
    print(f"增强提示词长度: {len(prompt)} 字符")
