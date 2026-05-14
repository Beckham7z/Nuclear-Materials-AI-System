"""
LangGraph Agent 节点定义
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from typing import Dict, Any, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
import json
import re

from .state import AgentState


# ============================================
# 1. 任务理解与分解节点
# ============================================
def task_understanding_node(state: AgentState) -> AgentState:
    """
    理解用户问题，分解任务
    """
    user_input = state["user_input"]
    
    # 添加思考过程
    thought = {
        "node": "task_understanding",
        "thought": f"正在理解用户问题: {user_input[:50]}...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 任务类型识别
    task_type = identify_task_type(user_input)
    
    # 分解任务步骤
    task_steps = decompose_task(user_input, task_type)
    
    return {
        **state,
        "current_task": task_type,
        "task_steps": task_steps,
        "agent_thoughts": state["agent_thoughts"] + [thought],
        "status": "running"
    }


def identify_task_type(user_input: str) -> str:
    """识别任务类型"""
    user_lower = user_input.lower()
    
    if any(kw in user_lower for kw in ["标准", "规范", "ASTM", "GB", "ISO"]):
        return "standard_matching"
    elif any(kw in user_lower for kw in ["对比", "比较", "优缺点"]):
        return "material_comparison"
    elif any(kw in user_lower for kw in ["性能", "特性", "强度", "耐腐蚀"]):
        return "performance_analysis"
    elif any(kw in user_lower for kw in ["最新", "文献", "论文", "研究"]):
        return "literature_review"
    elif any(kw in user_lower for kw in ["替代", "替换", "可选"]):
        return "material_substitution"
    elif any(kw in user_lower for kw in ["学习", "入门", "介绍", "什么是"]):
        return "learning_guidance"
    else:
        return "general_qa"


def decompose_task(user_input: str, task_type: str) -> list:
    """分解任务为步骤"""
    base_steps = [
        {"id": 1, "name": "问题理解", "description": "理解用户问题意图", "status": "completed"},
        {"id": 2, "name": "RAG检索", "description": "从知识库检索相关信息", "status": "pending"},
        {"id": 3, "name": "知识图谱查询", "description": "查询实体关系", "status": "pending"},
        {"id": 4, "name": "综合分析", "description": "整合检索结果进行分析", "status": "pending"},
        {"id": 5, "name": "生成回答", "description": "生成最终答案", "status": "pending"}
    ]
    
    # 根据任务类型添加特定步骤
    if task_type == "standard_matching":
        base_steps.insert(3, {"id": 3.5, "name": "标准匹配", "description": "匹配相关标准规范", "status": "pending"})
    elif task_type == "material_comparison":
        base_steps.insert(3, {"id": 3.5, "name": "材料对比", "description": "对比不同材料特性", "status": "pending"})
    
    return base_steps


# ============================================
# 2. RAG 检索节点
# ============================================
def rag_retrieval_node(state: AgentState) -> AgentState:
    """
    RAG 检索
    """
    thought = {
        "node": "rag_retrieval",
        "thought": "开始执行 RAG 检索...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 更新任务步骤
    task_steps = state["task_steps"].copy()
    for step in task_steps:
        if step["name"] == "RAG检索":
            step["status"] = "completed"
            step["result"] = f"检索到相关信息"
        elif step["name"] == "知识图谱查询" and step["status"] == "pending":
            step["status"] = "in_progress"
    
    return {
        **state,
        "task_steps": task_steps,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


# ============================================
# 3. 知识图谱查询节点
# ============================================
def kg_query_node(state: AgentState) -> AgentState:
    """
    知识图谱查询
    """
    thought = {
        "node": "kg_query",
        "thought": "查询知识图谱获取实体关系...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 更新任务步骤
    task_steps = state["task_steps"].copy()
    for step in task_steps:
        if step["name"] == "知识图谱查询":
            step["status"] = "completed"
            step["result"] = "获取到实体关系"
        elif step["name"] == "综合分析" and step["status"] == "pending":
            step["status"] = "in_progress"
    
    return {
        **state,
        "task_steps": task_steps,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


# ============================================
# 4. 综合分析节点
# ============================================
def synthesis_node(state: AgentState) -> AgentState:
    """
    综合分析检索结果
    """
    thought = {
        "node": "synthesis",
        "thought": "正在综合分析检索结果...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 更新任务步骤
    task_steps = state["task_steps"].copy()
    for step in task_steps:
        if step["name"] == "综合分析":
            step["status"] = "completed"
            step["result"] = "分析完成"
        elif step["name"] == "生成回答" and step["status"] == "pending":
            step["status"] = "in_progress"
    
    return {
        **state,
        "task_steps": task_steps,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


# ============================================
# 5. 答案生成节点
# ============================================
def answer_generation_node(state: AgentState) -> AgentState:
    """
    生成最终答案
    """
    thought = {
        "node": "answer_generation",
        "thought": "正在生成最终答案...",
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        **state,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


# ============================================
# 6. 人机协同节点
# ============================================
def human_feedback_node(state: AgentState) -> AgentState:
    """
    人机协同反馈
    """
    return {
        **state,
        "need_human_review": True,
        "status": "waiting_human",
        "agent_thoughts": state["agent_thoughts"] + [{
            "node": "human_feedback",
            "thought": "等待用户确认或补充信息...",
            "timestamp": datetime.now().isoformat()
        }]
    }


def process_human_feedback(state: AgentState, feedback: str) -> AgentState:
    """
    处理人类反馈
    """
    return {
        **state,
        "human_feedback": feedback,
        "need_human_review": False,
        "status": "running",
        "agent_thoughts": state["agent_thoughts"] + [{
            "node": "human_feedback",
            "thought": f"收到用户反馈: {feedback[:30]}...",
            "timestamp": datetime.now().isoformat()
        }]
    }


# ============================================
# 7. 自我反思节点
# ============================================
def self_reflection_node(state: AgentState) -> AgentState:
    """
    自我反思答案质量
    """
    thought = {
        "node": "self_reflection",
        "thought": "正在反思答案质量...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 反思结果
    reflection = {
        "quality_score": 0.85,
        "needs_improvement": False,
        "suggestions": [],
        "confidence": "high",
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        **state,
        "self_reflection": reflection,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


# ============================================
# 8. 知识更新节点
# ============================================
def knowledge_update_node(state: AgentState) -> AgentState:
    """
    更新知识库
    """
    thought = {
        "node": "knowledge_update",
        "thought": "检查是否需要更新知识库...",
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        **state,
        "knowledge_updated": True,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


# ============================================
# 9. 结束节点
# ============================================
def finish_node(state: AgentState) -> AgentState:
    """
    完成任务
    """
    thought = {
        "node": "finish",
        "thought": "任务完成",
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        **state,
        "status": "completed",
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


# ============================================
# 10. 错误处理节点
# ============================================
def error_node(state: AgentState, error_message: str) -> AgentState:
    """
    错误处理
    """
    return {
        **state,
        "error": error_message,
        "status": "error",
        "agent_thoughts": state["agent_thoughts"] + [{
            "node": "error",
            "thought": f"发生错误: {error_message}",
            "timestamp": datetime.now().isoformat()
        }]
    }
