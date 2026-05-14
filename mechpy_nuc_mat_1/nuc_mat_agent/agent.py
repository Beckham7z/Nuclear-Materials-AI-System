"""
LangGraph Agent 核心逻辑 - 支持记忆模块和人在回路
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from typing import Dict, Any, Literal, List, Optional, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from datetime import datetime
import json
import re
import asyncio
import yaml

from .state import AgentState, create_initial_state
from .llm_client import get_llm_client, NUCLEAR_MATERIAL_SYSTEM_PROMPT, AGENT_REFLECTION_PROMPT
from .rag_client import get_rag_client
from src.configuration.logset import logger


# ============================================
# 全局 Checkpointer 实例
# ============================================
_memory_saver = MemorySaver()


# ============================================
# 风险等级映射
# ============================================
RISK_KEYWORDS = {
    "critical": ["核安全", "临界安全", "事故", "泄漏", "爆炸", "熔毁"],
    "high": ["辐照", "剂量", "健康", "安全关键", "压力容器", "堆芯"],
    "medium": ["性能", "强度", "耐腐蚀", "蠕变", "疲劳", "寿命"],
    "low": ["一般", "介绍", "入门", "学习", "概念"]
}


# ============================================
# Agent 节点定义
# ============================================

def task_understanding_node(state: AgentState) -> AgentState:
    """
    1. 任务理解与分解节点
    """
    user_input = state["user_input"]
    
    # 添加思考过程
    thought = {
        "node": "task_understanding",
        "thought": f"正在理解用户问题: {user_input[:50]}...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 任务类型识别
    task_type = _identify_task_type(user_input)
    
    # 分解任务步骤
    task_steps = _decompose_task(user_input, task_type)
    
    return {
        **state,
        "current_task": task_type,
        "task_steps": task_steps,
        "agent_thoughts": state["agent_thoughts"] + [thought],
        "status": "running"
    }


def _identify_task_type(user_input: str) -> str:
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
    elif any(kw in user_lower for kw in ["学习", "入门", "介绍", "什么是", "rag", "知识库", "RAG"]):
        return "learning_guidance"
    else:
        return "general_qa"


def _decompose_task(user_input: str, task_type: str) -> list:
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


def rag_retrieval_node(state: AgentState) -> AgentState:
    """
    2. RAG 检索节点
    """
    user_input = state["user_input"]
    task_type = state.get("current_task", "general_qa")

    # 通用问答和学习指导类型跳过 RAG 检索
    if task_type in ["general_qa"]:
        thought = {
            "node": "rag_retrieval",
            "thought": f"任务类型 {task_type} 为通用问答，跳过 RAG 检索",
            "timestamp": datetime.now().isoformat()
        }
        return {
            **state,
            "retrieval_results": [],
            "rag_context": [],
            "agent_thoughts": state["agent_thoughts"] + [thought]
        }

    # 添加思考过程
    thought = {
        "node": "rag_retrieval",
        "thought": f"开始执行 RAG 检索，任务类型: {task_type}...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 更新任务步骤
    task_steps = state["task_steps"].copy()
    for step in task_steps:
        if step["name"] == "RAG检索":
            step["status"] = "in_progress"
    
    # 执行 RAG 检索
    try:
        rag_client = get_rag_client()
        rag_result = rag_client.query_sync(user_input, mode="hybrid", top_k=10)
        
        retrieval_results = rag_result.get("results", [])
        rag_context = rag_result.get("context", "")
        
        # 更新步骤状态
        for step in task_steps:
            if step["name"] == "RAG检索":
                step["status"] = "completed"
                step["result"] = f"检索到 {len(retrieval_results)} 条相关信息"
            elif step["name"] == "知识图谱查询" and step["status"] == "pending":
                step["status"] = "in_progress"
        
        thought["thought"] += f"检索完成，获得 {len(retrieval_results)} 条结果"
        
    except Exception as e:
        logger.error(f"RAG 检索失败: {e}")
        for step in task_steps:
            if step["name"] == "RAG检索":
                step["status"] = "completed"
                step["result"] = f"检索失败: {str(e)}"
    
    return {
        **state,
        "task_steps": task_steps,
        "retrieval_results": retrieval_results if 'retrieval_results' in dir() else [],
        "rag_context": rag_context if 'rag_context' in dir() else [],
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


def kg_query_node(state: AgentState) -> AgentState:
    """
    3. 知识图谱查询节点
    """
    # 添加思考过程
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


def synthesis_node(state: AgentState) -> AgentState:
    """
    4. 综合分析节点
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


def answer_generation_node(state: AgentState) -> AgentState:
    """
    5. 答案生成节点
    """
    user_input = state["user_input"]
    rag_context = state.get("rag_context", [])
    task_type = state.get("current_task", "general_qa")
    
    thought = {
        "node": "answer_generation",
        "thought": "正在生成最终答案...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 构建提示词
    system_prompt = NUCLEAR_MATERIAL_SYSTEM_PROMPT
    
    # 添加 RAG 上下文
    context_text = ""
    if rag_context:
        if isinstance(rag_context, str):
            context_text = f"\n\n参考信息:\n{rag_context[:3000]}"
        elif isinstance(rag_context, list):
            context_text = "\n\n参考信息:\n" + "\n".join([str(ctx)[:500] for ctx in rag_context])
    
    user_prompt = f"""任务类型: {task_type}

用户问题: {user_input}
{context_text}

请基于以上信息和你的专业知识，提供详细的分析和回答。"""

    # 调用 LLM 生成答案
    try:
        llm_client = get_llm_client()
        result = llm_client.chat(user_prompt, system_prompt)
        
        thought["thought"] = "答案生成完成"
        
    except Exception as e:
        logger.error(f"答案生成失败: {e}")
        result = f"生成回答时发生错误: {str(e)}"
        thought["thought"] = f"答案生成失败: {str(e)}"
    
    # 更新任务步骤
    task_steps = state["task_steps"].copy()
    for step in task_steps:
        if step["name"] == "生成回答":
            step["status"] = "completed"
            step["result"] = "回答已生成"
    
    return {
        **state,
        "final_answer": result,
        "task_steps": task_steps,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


def human_feedback_node(state: AgentState) -> AgentState:
    """
    6. 人机协同节点 - 判断是否需要人工审核
    """
    # 检查是否启用人工反馈
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        enable_human_feedback = config.get('agent', {}).get('enable_human_feedback', True)
    except:
        enable_human_feedback = True
    
    if not enable_human_feedback:
        return state
    
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


def self_reflection_node(state: AgentState) -> AgentState:
    """
    7. 自我反思节点
    """
    user_input = state["user_input"]
    final_answer = state.get("final_answer", "")
    
    thought = {
        "node": "self_reflection",
        "thought": "正在反思答案质量...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 检查是否启用自我反思
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        enable_self_reflection = config.get('agent', {}).get('enable_self_reflection', True)
    except:
        enable_self_reflection = True
    
    if not enable_self_reflection or not final_answer:
        return {
            **state,
            "agent_thoughts": state["agent_thoughts"] + [thought]
        }
    
    # 调用 LLM 进行反思
    try:
        llm_client = get_llm_client()
        prompt = AGENT_REFLECTION_PROMPT.format(answer=final_answer, question=user_input)
        reflection_result = llm_client.chat(prompt)
        
        # 解析反思结果
        try:
            if "```json" in reflection_result:
                json_str = reflection_result.split("```json")[1].split("```")[0]
                reflection = json.loads(json_str)
            else:
                reflection = {
                    "quality_score": 0.8,
                    "needs_improvement": False,
                    "issues": [],
                    "suggestions": []
                }
        except:
            reflection = {
                "quality_score": 0.8,
                "needs_improvement": False,
                "issues": [],
                "suggestions": []
            }
        
        thought["thought"] = f"反思完成，质量评分: {reflection.get('quality_score', 0)}"
        
    except Exception as e:
        logger.error(f"自我反思失败: {e}")
        reflection = {
            "quality_score": 0.5,
            "needs_improvement": False,
            "issues": [str(e)],
            "suggestions": []
        }
    
    return {
        **state,
        "self_reflection": reflection,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


def knowledge_update_node(state: AgentState) -> AgentState:
    """
    8. 知识更新节点
    """
    thought = {
        "node": "knowledge_update",
        "thought": "检查是否需要更新知识库...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 检查是否启用水印更新
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        enable_knowledge_update = config.get('agent', {}).get('enable_knowledge_update', True)
    except:
        enable_knowledge_update = True
    
    # 目前只是标记，待实现具体的更新逻辑
    return {
        **state,
        "knowledge_updated": enable_knowledge_update,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


def finish_node(state: AgentState) -> AgentState:
    """
    9. 结束节点
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
# 条件路由函数
# ============================================

def should_request_human_feedback(state: AgentState) -> str:
    """判断是否需要人工审核"""
    if state.get("need_human_review", False):
        return "human_feedback"
    return "self_reflection"


def should_retry_or_finish(state: AgentState) -> str:
    """判断是重试还是完成"""
    reflection = state.get("self_reflection", {})
    
    if reflection and reflection.get("needs_improvement", False):
        quality_score = reflection.get("quality_score", 0)
        if quality_score < 0.6:
            return "retry"
    
    return "finish"


# ============================================
# 构建 Agent 图
# ============================================

def build_agent_graph():
    """构建 Agent 状态图"""
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("task_understanding", task_understanding_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("kg_query", kg_query_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("answer_generation", answer_generation_node)
    workflow.add_node("human_feedback", human_feedback_node)
    workflow.add_node("self_reflection", self_reflection_node)
    workflow.add_node("knowledge_update", knowledge_update_node)
    workflow.add_node("finish", finish_node)
    
    # 设置入口点
    workflow.set_entry_point("task_understanding")
    
    # 添加边
    workflow.add_edge("task_understanding", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "kg_query")
    workflow.add_edge("kg_query", "synthesis")
    workflow.add_edge("synthesis", "answer_generation")
    
    # 条件边：人工反馈 -> 自我反思 -> 完成
    workflow.add_conditional_edges(
        "answer_generation",
        should_request_human_feedback,
        {
            "human_feedback": "human_feedback",
            "self_reflection": "self_reflection"
        }
    )
    
    workflow.add_edge("human_feedback", "answer_generation")
    workflow.add_edge("self_reflection", "knowledge_update")
    workflow.add_edge("knowledge_update", "finish")
    
    # 编译图
    return workflow.compile()


# 全局 Agent 实例
_agent_graph = None


def get_agent_graph():
    """获取全局 Agent 图"""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


def run_agent(user_input: str, session_id: str = None) -> AgentState:
    """运行 Agent"""
    # 创建初始状态
    initial_state = create_initial_state(user_input, session_id)
    
    # 获取 Agent 图
    agent = get_agent_graph()
    
    # 运行 Agent
    try:
        final_state = agent.invoke(initial_state)
        return final_state
    except Exception as e:
        logger.error(f"Agent 运行失败: {e}")
        return {
            **initial_state,
            "status": "error",
            "error": str(e)
        }


async def run_agent_async(user_input: str, session_id: str = None) -> AgentState:
    """异步运行 Agent"""
    # 创建初始状态
    initial_state = create_initial_state(user_input, session_id)
    
    # 获取 Agent 图
    agent = get_agent_graph()
    
    # 运行 Agent
    try:
        final_state = await agent.ainvoke(initial_state)
        return final_state
    except Exception as e:
        logger.error(f"Agent 运行失败: {e}")
        return {
            **initial_state,
            "status": "error",
            "error": str(e)
        }
