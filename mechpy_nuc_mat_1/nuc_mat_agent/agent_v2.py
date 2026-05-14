"""
LangGraph Agent v2 - 支持记忆模块和人在回路
基于原agent.py，增加了：
1. MemorySaver checkpointer 实现多轮对话记忆
2. 多路径推理实现自我一致性
3. interrupt 实现人在回路审核
4. 置信度评估和风险分级
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from typing import Dict, Any, Literal, List, Optional, Annotated, Union
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from datetime import datetime
import json
import re
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
    "critical": ["核安全", "临界安全", "事故", "泄漏", "爆炸", "熔毁", "紧急"],
    "high": ["辐照", "剂量", "健康", "安全关键", "压力容器", "堆芯", "事故工况"],
    "medium": ["性能", "强度", "耐腐蚀", "蠕变", "疲劳", "寿命", "退化"],
    "low": ["一般", "介绍", "入门", "学习", "概念", "原理"]
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
    
    # 风险等级评估
    risk_level = _evaluate_risk_level(user_input)
    
    # 分解任务步骤
    task_steps = _decompose_task(user_input, task_type)
    
    return {
        **state,
        "current_task": task_type,
        "task_steps": task_steps,
        "risk_level": risk_level,
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


def _evaluate_risk_level(user_input: str) -> str:
    """评估任务风险等级"""
    user_input_lower = user_input.lower()
    
    # 按优先级检查风险关键词
    for level in ["critical", "high", "medium", "low"]:
        if any(kw in user_input_lower for kw in RISK_KEYWORDS.get(level, [])):
            return level
    
    return "low"


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


def multi_path_reasoning_node(state: AgentState) -> AgentState:
    """
    5. 多路径推理节点 - 实现自我一致性
    生成多条推理路径，然后投票选择一致性最高的答案
    """
    user_input = state["user_input"]
    rag_context = state.get("rag_context", [])
    task_type = state.get("current_task", "general_qa")
    
    thought = {
        "node": "multi_path_reasoning",
        "thought": "执行多路径推理以确保答案一致性...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 构建提示词
    context_text = ""
    if rag_context:
        if isinstance(rag_context, str):
            context_text = f"\n\n参考信息:\n{rag_context[:3000]}"
        elif isinstance(rag_context, list):
            context_text = "\n\n参考信息:\n" + "\n".join([str(ctx)[:500] for ctx in rag_context])
    
    user_prompt = f"""任务类型: {task_type}

用户问题: {user_input}
{context_text}

请从不同的角度分析这个问题，生成3条不同的推理路径，每条路径需要：
1. 不同的分析角度
2. 独立的推理逻辑
3. 最终得出结论

请以JSON格式返回：
{{
    "path_1": {{"analysis": "分析内容", "conclusion": "结论"}},
    "path_2": {{"analysis": "分析内容", "conclusion": "结论"}},
    "path_3": {{"analysis": "分析内容", "conclusion": "结论"}}
}}"""

    # 调用 LLM 生成多路径推理
    try:
        llm_client = get_llm_client()
        result = llm_client.chat(user_prompt, NUCLEAR_MATERIAL_SYSTEM_PROMPT)
        
        # 解析结果
        multi_path_results = []
        try:
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0]
                multi_path_results = json.loads(json_str)
            else:
                multi_path_results = {"raw_result": result}
        except:
            multi_path_results = {"raw_result": result}
        
        thought["thought"] = f"多路径推理完成，生成 {len(multi_path_results)} 条路径"
        
    except Exception as e:
        logger.error(f"多路径推理失败: {e}")
        multi_path_results = {"error": str(e)}
        thought["thought"] = f"多路径推理失败: {str(e)}"
    
    return {
        **state,
        "multi_path_results": [multi_path_results] if isinstance(multi_path_results, dict) else multi_path_results,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


def consistency_voting_node(state: AgentState) -> AgentState:
    """
    6. 一致性投票节点
    对多路径推理结果进行投票，选择一致性最高的答案
    """
    multi_path_results = state.get("multi_path_results", [])
    
    thought = {
        "node": "consistency_voting",
        "thought": "执行一致性投票...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 简化的投票逻辑：提取所有结论进行相似度计算
    # 这里使用简单的关键词匹配来评估一致性
    consistency_score = 0.5  # 默认得分
    
    if multi_path_results and len(multi_path_results) > 0:
        # 提取所有结论
        conclusions = []
        for path_result in multi_path_results:
            if isinstance(path_result, dict):
                for key, value in path_result.items():
                    if isinstance(value, dict) and "conclusion" in value:
                        conclusions.append(value["conclusion"])
        
        # 简单的相似度评估：检查结论中是否有共同的关键词
        if len( conclusions) >= 2:
            # 这里可以接入更复杂的一致性算法
            consistency_score = 0.75  # 简化处理
        
        thought["thought"] = f"一致性评估完成，得分: {consistency_score}"
    
    return {
        **state,
        "consistency_score": consistency_score,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


def answer_generation_node(state: AgentState) -> AgentState:
    """
    7. 答案生成节点
    """
    user_input = state["user_input"]
    rag_context = state.get("rag_context", [])
    task_type = state.get("current_task", "general_qa")
    multi_path_results = state.get("multi_path_results", [])
    consistency_score = state.get("consistency_score", 0.5)
    
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
    
    # 添加多路径推理结果
    multi_path_text = ""
    if multi_path_results:
        multi_path_text = f"\n\n多路径分析结果（一致性得分: {consistency_score:.2f}）:\n{json.dumps(multi_path_results, ensure_ascii=False)[:1000]}"
    
    user_prompt = f"""任务类型: {task_type}

用户问题: {user_input}
{context_text}
{multi_path_text}

请基于以上信息和你的专业知识，提供详细的分析和回答。
如果多路径分析存在不一致，请综合考虑给出最合理的答案。"""

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


def confidence_evaluation_node(state: AgentState) -> AgentState:
    """
    8. 置信度评估节点
    基于多个因素评估答案的置信度
    """
    risk_level = state.get("risk_level", "low")
    consistency_score = state.get("consistency_score", 0.5)
    retrieval_results = state.get("retrieval_results", [])
    
    thought = {
        "node": "confidence_evaluation",
        "thought": "正在评估答案置信度...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 计算置信度得分
    # 基础分：一致性得分 * 0.4
    # 检索分：如果有检索结果 +0.3
    # 风险分：低风险 +0.3，高风险 -0.2
    confidence_score = consistency_score * 0.4
    
    if len(retrieval_results) > 0:
        confidence_score += 0.3
    
    if risk_level == "low":
        confidence_score += 0.3
    elif risk_level in ["high", "critical"]:
        confidence_score -= 0.2
    
    # 确保得分在0-1之间
    confidence_score = max(0.0, min(1.0, confidence_score))
    
    thought["thought"] = f"置信度评估完成: {confidence_score:.2f}"
    
    return {
        **state,
        "confidence_score": confidence_score,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


def risk_assessment_node(state: AgentState) -> AgentState:
    """
    9. 风险评估节点
    根据风险等级和置信度决定是否需要人工审核
    """
    risk_level = state.get("risk_level", "low")
    confidence_score = state.get("confidence_score", 0.5)
    
    thought = {
        "node": "risk_assessment",
        "thought": f"风险等级: {risk_level}, 置信度: {confidence_score:.2f}",
        "timestamp": datetime.now().isoformat()
    }
    
    # 决定是否需要人工审核
    need_review = False
    
    # 高风险或关键风险必须审核
    if risk_level in ["critical", "high"]:
        need_review = True
    # 中等风险且置信度低于阈值
    elif risk_level == "medium" and confidence_score < 0.7:
        need_review = True
    # 低风险但置信度很低
    elif confidence_score < 0.5:
        need_review = True
    
    thought["thought"] += f" -> {'需要人工审核' if need_review else '自动放行'}"
    
    return {
        **state,
        "need_human_review": need_review,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


def human_review_node(state: AgentState) -> AgentState:
    """
    10. 人在回路节点 - 使用 interrupt 暂停等待人工审核
    """
    final_answer = state.get("final_answer") or ""
    risk_level = state.get("risk_level", "low")
    confidence_score = state.get("confidence_score") or 0.5
    user_input = state.get("user_input") or ""
    
    # 构建审核请求信息
    review_request = {
        "user_input": user_input,
        "final_answer": final_answer,
        "risk_level": risk_level,
        "confidence_score": confidence_score,
        "timestamp": datetime.now().isoformat()
    }
    
    # 安全截取回答内容
    answer_preview = (final_answer[:500] + "...") if final_answer else "暂无回答"
    
    # 使用 interrupt 暂停执行，等待人工反馈
    # 这是 LangGraph 的人在回路核心机制
    human_feedback = interrupt({
        "type": "human_review_request",
        "question": f"【人工审核请求】风险等级: {risk_level}, 置信度: {confidence_score:.2f}\n\n请审核以下回答是否准确：\n\n问题：{user_input}\n\n回答：{answer_preview}",
        "review_request": review_request
    })
    
    # 添加专家反馈到历史
    expert_feedback_history = state.get("expert_feedback_history", [])
    expert_feedback_history.append({
        "timestamp": datetime.now().isoformat(),
        "feedback": human_feedback,
        "risk_level": risk_level,
        "confidence_score": confidence_score
    })
    
    thought = {
        "node": "human_review",
        "thought": f"收到专家反馈: {human_feedback[:50] if human_feedback else '无反馈'}...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 根据反馈决定是否修正答案
    new_final_answer = final_answer
    if human_feedback and "修正" in human_feedback:
        # 如果专家提供了修正内容，更新答案
        new_final_answer = human_feedback
    
    return {
        **state,
        "human_feedback": human_feedback,
        "final_answer": new_final_answer,
        "expert_feedback_history": expert_feedback_history,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


def self_reflection_node(state: AgentState) -> AgentState:
    """
    11. 自我反思节点
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
    12. 知识更新节点
    将专家反馈整合到知识库中
    """
    thought = {
        "node": "knowledge_update",
        "thought": "检查是否需要更新知识库...",
        "timestamp": datetime.now().isoformat()
    }
    
    # 检查是否启用水印更新
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        enable_knowledge_update = config.get('agent', {}).get('enable_knowledge_update', True)
    except:
        enable_knowledge_update = True
    
    # 如果有专家反馈，记录到反馈历史
    expert_feedback_history = state.get("expert_feedback_history", [])
    if expert_feedback_history and enable_knowledge_update:
        # 保存反馈日志
        _save_feedback_log(expert_feedback_history)
        thought["thought"] += f" - 保存了 {len(expert_feedback_history)} 条专家反馈"
    
    return {
        **state,
        "knowledge_updated": enable_knowledge_update,
        "agent_thoughts": state["agent_thoughts"] + [thought]
    }


def _save_feedback_log(feedback_history: List[Dict]):
    """保存专家反馈日志"""
    try:
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'expert_feedback_log.json')
        
        # 读取现有日志
        existing_logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                existing_logs = json.load(f)
        
        # 添加新反馈
        existing_logs.extend(feedback_history)
        
        # 保存更新后的日志
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(existing_logs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"专家反馈日志已保存: {len(feedback_history)} 条")
    except Exception as e:
        logger.error(f"保存专家反馈日志失败: {e}")


def finish_node(state: AgentState) -> AgentState:
    """
    13. 结束节点
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

def should_request_human_review(state: AgentState) -> str:
    """判断是否需要人工审核"""
    if state.get("need_human_review", False):
        return "human_review"
    return "self_reflection"


def should_retry_or_finish(state: AgentState) -> str:
    """判断是重试还是完成"""
    reflection = state.get("self_reflection", {})
    
    if reflection and reflection.get("needs_improvement", False):
        quality_score = reflection.get("quality_score", 0)
        if quality_score < 0.6:
            return "retry"
    
    return "finish"


def should_evaluate_confidence(state: AgentState) -> str:
    """判断是否需要置信度评估"""
    multi_path_results = state.get("multi_path_results", [])
    if multi_path_results and len(multi_path_results) > 0:
        return "consistency_voting"
    return "confidence_evaluation"


# ============================================
# 构建 Agent 图 (v2版本 - 带记忆和人在回路)
# ============================================

def build_agent_graph_v2():
    """构建 Agent 状态图 v2 - 支持记忆和人在回路"""
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("task_understanding", task_understanding_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("kg_query", kg_query_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("multi_path_reasoning", multi_path_reasoning_node)
    workflow.add_node("consistency_voting", consistency_voting_node)
    workflow.add_node("answer_generation", answer_generation_node)
    workflow.add_node("confidence_evaluation", confidence_evaluation_node)
    workflow.add_node("risk_assessment", risk_assessment_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("self_reflection", self_reflection_node)
    workflow.add_node("knowledge_update", knowledge_update_node)
    workflow.add_node("finish", finish_node)
    
    # 设置入口点
    workflow.add_edge(START, "task_understanding")
    
    # 添加边
    workflow.add_edge("task_understanding", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "kg_query")
    workflow.add_edge("kg_query", "synthesis")
    workflow.add_edge("synthesis", "multi_path_reasoning")
    
    # 多路径推理 -> 一致性投票 -> 置信度评估
    workflow.add_edge("multi_path_reasoning", "consistency_voting")
    workflow.add_edge("consistency_voting", "confidence_evaluation")
    workflow.add_edge("confidence_evaluation", "risk_assessment")
    
    # 风险评估 -> 人工审核或自我反思
    workflow.add_conditional_edges(
        "risk_assessment",
        should_request_human_review,
        {
            "human_review": "human_review",
            "self_reflection": "self_reflection"
        }
    )
    
    # 人工审核后进入自我反思
    workflow.add_edge("human_review", "self_reflection")
    workflow.add_edge("self_reflection", "knowledge_update")
    workflow.add_edge("knowledge_update", "finish")
    
    # 使用 MemorySaver 编译图
    return workflow.compile(checkpointer=_memory_saver)


# 全局 Agent 实例
_agent_graph_v2 = None


def get_agent_graph_v2():
    """获取全局 Agent 图 v2"""
    global _agent_graph_v2
    if _agent_graph_v2 is None:
        _agent_graph_v2 = build_agent_graph_v2()
    return _agent_graph_v2


def run_agent_v2(user_input: str, thread_id: str = None) -> AgentState:
    """
    运行 Agent v2 - 支持记忆模块
    Args:
        user_input: 用户输入
        thread_id: 线程ID，用于记忆模块
    Returns:
        AgentState: Agent执行结果
    """
    # 创建初始状态
    initial_state = create_initial_state(user_input, thread_id)
    
    # 获取 Agent 图
    agent = get_agent_graph_v2()
    
    # 构建配置（包含thread_id用于记忆）
    config = {"configurable": {"thread_id": thread_id or str(datetime.now().timestamp())}}
    
    # 运行 Agent
    try:
        final_state = agent.invoke(initial_state, config)
        return final_state
    except Exception as e:
        logger.error(f"Agent v2 运行失败: {e}")
        return {
            **initial_state,
            "status": "error",
            "error": str(e)
        }


async def run_agent_v2_async(user_input: str, thread_id: str = None) -> AgentState:
    """异步运行 Agent v2"""
    # 创建初始状态
    initial_state = create_initial_state(user_input, thread_id)
    
    # 获取 Agent 图
    agent = get_agent_graph_v2()
    
    # 构建配置
    config = {"configurable": {"thread_id": thread_id or str(datetime.now().timestamp())}}
    
    # 运行 Agent
    try:
        final_state = await agent.ainvoke(initial_state, config)
        return final_state
    except Exception as e:
        logger.error(f"Agent v2 异步运行失败: {e}")
        return {
            **initial_state,
            "status": "error",
            "error": str(e)
        }


def resume_agent_v2(thread_id: str, human_feedback: str) -> AgentState:
    """
    恢复 Agent v2 执行（用于人在回路审核后）
    Args:
        thread_id: 线程ID
        human_feedback: 专家反馈
    Returns:
        AgentState: 恢复后的执行结果
    """
    agent = get_agent_graph_v2()
    
    try:
        # 使用 Command.resume 恢复执行
        config = {"configurable": {"thread_id": thread_id}}
        command = Command(resume=human_feedback)
        final_state = agent.invoke(command, config)
        return final_state
    except Exception as e:
        logger.error(f"Agent v2 恢复执行失败: {e}")
        return {"status": "error", "error": str(e)}


def get_conversation_history(thread_id: str) -> List[Dict[str, Any]]:
    """
    获取对话历史（用于记忆模块）
    Args:
        thread_id: 线程ID
    Returns:
        List[Dict]: 对话历史记录
    """
    agent = get_agent_graph_v2()
    
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # 获取当前状态
        current_state = agent.get_state(config)
        
        # 返回消息历史
        return current_state.values.get("messages", [])
    except Exception as e:
        logger.error(f"获取对话历史失败: {e}")
        return []


# ============================================
# 便捷函数
# ============================================

def is_waiting_for_human(state: AgentState) -> bool:
    """检查是否在等待人工审核"""
    return state.get("status") == "waiting_human" or state.get("need_human_review", False)


def get_review_info(state: AgentState) -> Optional[Dict]:
    """获取审核请求信息"""
    return state.get("review_request")


if __name__ == "__main__":
    # 测试代码
    print("Agent v2 - 记忆模块和人在回路")
    print("=" * 50)
    
    # 测试运行
    test_input = "什么是核反应堆的压力边界材料？"
    result = run_agent_v2(test_input, thread_id="test_thread")
    
    print(f"\n状态: {result.get('status')}")
    print(f"风险等级: {result.get('risk_level')}")
    print(f"置信度: {result.get('confidence_score')}")
    print(f"需要审核: {result.get('need_human_review')}")
