"""
LangGraph Agent 状态定义
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import add_messages
from datetime import datetime
import uuid


class AgentState(TypedDict):
    """Agent 状态定义"""
    # 对话历史 (用于记忆模块)
    messages: Annotated[List[Dict[str, Any]], add_messages]
    
    # 用户输入
    user_input: str
    
    # 当前任务
    current_task: str
    
    # 任务步骤
    task_steps: List[Dict[str, Any]]
    
    # 检索结果
    retrieval_results: List[Dict[str, Any]]
    
    # 知识图谱结果
    kg_results: List[Dict[str, Any]]
    
    # RAG 上下文
    rag_context: List[str]
    
    # Agent 思考过程
    agent_thoughts: List[Dict[str, str]]
    
    # 中间答案
    intermediate_answers: List[Dict[str, Any]]
    
    # 最终答案
    final_answer: Optional[str]
    
    # 人机协同反馈
    human_feedback: Optional[str]
    
    # 是否需要人机协同
    need_human_review: bool
    
    # 自我反思结果
    self_reflection: Optional[Dict[str, Any]]
    
    # 知识更新标记
    knowledge_updated: bool
    
    # 错误信息
    error: Optional[str]
    
    # 执行状态
    status: str  # "idle", "running", "waiting_human", "completed", "error"
    
    # 时间戳
    timestamp: str
    
    # 会话ID
    session_id: str
    
    # ========== 新增字段 ==========
    # 置信度得分 (0.0 - 1.0)
    confidence_score: Optional[float]
    
    # 风险等级: "low", "medium", "high", "critical"
    risk_level: str
    
    # 多路径推理结果 (用于自我一致性)
    multi_path_results: List[Dict[str, Any]]
    
    # 一致性得分
    consistency_score: Optional[float]
    
    # 人工审核暂停信息
    review_request: Optional[Dict[str, Any]]
    
    # 专家反馈历史
    expert_feedback_history: List[Dict[str, Any]]


def create_initial_state(user_input: str, session_id: str = None) -> AgentState:
    """创建初始状态"""
    return AgentState(
        messages=[],
        user_input=user_input,
        current_task="",
        task_steps=[],
        retrieval_results=[],
        kg_results=[],
        rag_context=[],
        agent_thoughts=[],
        intermediate_answers=[],
        final_answer=None,
        human_feedback=None,
        need_human_review=False,
        self_reflection=None,
        knowledge_updated=False,
        error=None,
        status="idle",
        timestamp=datetime.now().isoformat(),
        session_id=session_id or str(uuid.uuid4()),
        # ========== 新增字段默认值 ==========
        confidence_score=None,
        risk_level="low",
        multi_path_results=[],
        consistency_score=None,
        review_request=None,
        expert_feedback_history=[]
    )
