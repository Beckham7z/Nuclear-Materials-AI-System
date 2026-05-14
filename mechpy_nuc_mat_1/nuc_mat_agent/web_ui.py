"""
Nuclear Material Agent - Streamlit 前端界面
左侧栏：配置 + 聊天记录
主体：Agent 聊天界面 + 过程展示
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import yaml
import json
import time
from datetime import datetime
from typing import Dict, Any, List

from .agent import run_agent, get_agent_graph
from .llm_client import get_llm_client
from .rag_client import get_rag_client
from .state import AgentState, create_initial_state

# 页面配置
st.set_page_config(
    page_title="NucmatPilot - 核电材料分析助手",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
    /* 主样式 */
    .main-header {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #1e3a5f;
        margin-bottom: 1rem;
    }
    
    /* 任务步骤样式 */
    .task-step {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid #3c6382;
    }
    
    .task-step.completed {
        border-left-color: #28a745;
        background: #e8f5e9;
    }
    
    .task-step.in_progress {
        border-left-color: #ffc107;
        background: #fff8e1;
    }
    
    .task-step.pending {
        border-left-color: #6c757d;
        opacity: 0.7;
    }
    
    /* 思考过程样式 */
    .thought-process {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-family: monospace;
        font-size: 0.9em;
    }
    
    .thought-node {
        padding: 4px 8px;
        margin: 4px 0;
        background: white;
        border-radius: 4px;
        border-left: 3px solid #2196f3;
    }
    
    /* 检索结果卡片 */
    .retrieval-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* 答案展示区 */
    .answer-section {
        background: #fafafa;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        border: 1px solid #e0e0e0;
    }
    
    /* 聊天消息样式 - 用户消息 */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* 聊天消息样式 - Agent 消息 */
    .assistant-message {
        background: linear-gradient(135deg, #e0eafc 0%, #f5f7fa 100%);
        border: 2px solid #3c6382;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(60, 99, 130, 0.15);
    }
    
    /* 消息气泡容器 */
    .chat-bubble-container {
        padding: 10px 0;
    }
    
    /* 用户消息气泡 */
    .user-bubble {
        display: flex;
        justify-content: flex-end;
        margin: 10px 0;
    }
    
    /* Agent 消息气泡 */
    .assistant-bubble {
        display: flex;
        justify-content: flex-start;
        margin: 10px 0;
    }
    
    /* 消息标签样式 */
    .message-label {
        font-size: 0.75em;
        font-weight: bold;
        margin-bottom: 4px;
        opacity: 0.8;
    }
    
    .user-message .message-label {
        text-align: right;
    }
    
    .assistant-message .message-label {
        text-align: left;
    }
    
    /* 配置面板 */
    .config-section {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    
    /* 状态指示器 */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: bold;
    }
    
    .status-running { background: #fff3cd; color: #856404; }
    .status-completed { background: #d4edda; color: #155724; }
    .status-error { background: #f8d7da; color: #721c24; }
    .status-waiting { background: #cce5ff; color: #004085; }
    
    /* 进度条 */
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #3c6382, #1e3799);
        transition: width 0.3s ease;
    }
    
    /* 侧边栏样式 */
    .sidebar-section {
        background: white;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# 初始化 Session State
# ============================================
def init_session_state():
    """初始化会话状态"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "current_agent_state" not in st.session_state:
        st.session_state.current_agent_state = None
    
    if "config" not in st.session_state:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                st.session_state.config = yaml.safe_load(f)
        except:
            st.session_state.config = {}
    
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())


init_session_state()


# ============================================
# 侧边栏 - 配置和聊天记录
# ============================================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("## ⚛️ NucmatPilot")
        
        # 模型配置
        st.markdown("### 🔧 模型配置")
        with st.expander("LLM 设置", expanded=True):
            config = st.session_state.config
            
            # LLM 提供商选择
            llm_options = ["minimax", "deepseek", "ollama", "zhipu"]
            default_idx = llm_options.index("minimax") if "minimax" in llm_options else 0
            
            selected_provider = st.selectbox(
                "LLM 提供商",
                llm_options,
                index=default_idx,
                help="选择 LLM 服务提供商"
            )
            
            # 获取提供商配置
            llm_config = config.get('llm', {}).get(selected_provider, {})
            current_model = llm_config.get('model', '')
            
            st.text_input(
                "模型名称",
                value=current_model,
                key="llm_model_display",
                disabled=True
            )
            
            # 显示 API 状态
            if llm_config.get('api_key'):
                st.success("✓ API Key 已配置")
            else:
                st.warning("⚠ API Key 未配置")
        
        # Agent 功能开关
        st.markdown("### 🤖 Agent 功能")
        with st.expander("功能配置", expanded=True):
            agent_config = config.get('agent', {})
            
            enable_human = st.toggle(
                "人机协同",
                value=agent_config.get('enable_human_feedback', True),
                help="开启后，Agent 会在生成答案前请求人工确认"
            )
            
            enable_reflection = st.toggle(
                "自我反思",
                value=agent_config.get('enable_self_reflection', True),
                help="开启后，Agent 会自动评估答案质量"
            )
            
            enable_update = st.toggle(
                "知识更新",
                value=agent_config.get('enable_knowledge_update', True),
                help="开启后，系统会记录高质量问答用于知识更新"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=agent_config.get('temperature', 1.0),
                step=0.1,
                help="控制回答的随机性"
            )
        
        # RAG 配置
        st.markdown("### 📚 RAG 配置")
        with st.expander("RAG 设置", expanded=False):
            rag_config = config.get('rag', {})

            st.text_input(
                "工作目录",
                value=rag_config.get('working_dir', '/home/zyx/A_project/mechpy_nuc_mat/myKG'),
                key="rag_working_dir_display",
                disabled=True
            )
            
            top_k = st.slider(
                "检索数量",
                min_value=1,
                max_value=30,
                value=rag_config.get('top_k', 10),
                step=1,
                help="RAG 检索返回的结果数量"
            )
            
            # 显示 RAG 状态
            try:
                rag_client = get_rag_client()
                if rag_client._initialized:
                    st.success("✓ RAG 已初始化")
                else:
                    st.info("RAG 待初始化")
            except:
                st.warning("RAG 未配置")
        
        st.divider()
        
        # 聊天记录
        st.markdown("### 💬 聊天记录")
        
        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # 显示历史记录列表
        for i, chat in enumerate(st.session_state.chat_history):
            if st.button(
                f"**{chat['time'][:16]}**\n{chat['question'][:30]}...",
                key=f"chat_{i}",
                use_container_width=True
            ):
                st.session_state.current_agent_state = chat.get('state')
                st.rerun()
        
        st.divider()
        
        # 统计信息
        st.markdown("### 📊 统计")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("对话数", len(st.session_state.chat_history))
        with col2:
            model_info = get_llm_client().get_model_info()
            st.metric("模型", model_info.get('model', 'N/A')[:7])


# ============================================
# 主界面 - Agent 聊天和过程展示
# ============================================
def render_main():
    """渲染主界面"""
    
    # 标题
    st.markdown('<p class="main-header" style="font-size: 2em;"> NucmatPilot - 核电材料分析助手</p>', unsafe_allow_html=True)
    
    # 任务说明
    with st.expander("README", expanded=False):
        st.info("""
        💡 这是一个基于 LangGraph 的智能 Agent，支持多步推理、人机协同、自我反思和知识更新。
        它可以帮你分析核电材料相关问题、检索文献、匹配标准、对比材料等。
        """)
    
    # 创建布局：左侧 Agent 过程，右侧聊天
    col_chat, col_process= st.columns([4, 1])
    
    with col_process:
        render_agent_process()
    
    with col_chat:
        render_chat_area()


def render_agent_process():
    """渲染 Agent 执行过程"""
    st.markdown('<h3 style="font-size: 1.1em;">🔄 Agent 执行过程</h3>', unsafe_allow_html=True)
    
    # 获取当前 Agent 状态
    current_state = st.session_state.current_agent_state
    
    if current_state is None:
        return
    
    # 1. 任务理解与分解
    st.markdown("#### 📋 任务分解")
    
    task_steps = current_state.get("task_steps", [])
    current_task = current_state.get("current_task", "unknown")
    
    # 任务类型标签
    task_labels = {
        "standard_matching": "📜 标准匹配",
        "material_comparison": "⚖️ 材料对比",
        "performance_analysis": "📊 性能分析",
        "literature_review": "📚 文献综述",
        "material_substitution": "🔄 材料替代",
        "learning_guidance": "📖 学习指导",
        "general_qa": "💬 通用问答"
    }
    
    task_label = task_labels.get(current_task, current_task)
    st.markdown(f"**任务类型**: {task_label}")
    
    # 显示任务步骤进度
    completed_steps = sum(1 for s in task_steps if s.get("status") == "completed")
    total_steps = len(task_steps)
    progress = completed_steps / total_steps if total_steps > 0 else 0
    
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress*100}%"></div>
    </div>
    <small>{completed_steps}/{total_steps} 步骤完成</small>
    """, unsafe_allow_html=True)
    
    # 显示每个步骤
    for step in task_steps:
        status = step.get("status", "pending")
        name = step.get("name", "")
        description = step.get("description", "")
        result = step.get("result", "")
        
        status_class = {
            "completed": "completed",
            "in_progress": "in_progress",
            "pending": "pending"
        }.get(status, "pending")
        
        status_icon = {
            "completed": "✅",
            "in_progress": "⏳",
            "pending": "⭕"
        }.get(status, "⭕")
        
        with st.container():
            st.markdown(f"""
            <div class="task-step {status_class}">
                <div>{status_icon} <strong>{name}</strong></div>
                <small>{description}</small>
                {f"<br><small style='color: green;'>✓ {result}</small>" if result else ""}
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # 2. 思考过程
    st.markdown("#### 🧠 Agent 思考过程")
    
    agent_thoughts = current_state.get("agent_thoughts", [])
    
    for i, thought in enumerate(agent_thoughts):
        node = thought.get("node", "")
        content = thought.get("thought", "")
        timestamp = thought.get("timestamp", "")[11:19]  # 只显示时间
        
        with st.expander(f"**{timestamp}** - {node}", expanded=(i == len(agent_thoughts) - 1)):
            st.markdown(f"<div class='thought-node'>{content}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # 3. 检索结果
    st.markdown("#### 🔍 RAG 检索结果")
    
    retrieval_results = current_state.get("retrieval_results", [])
    
    if not retrieval_results:
        st.info("暂无检索结果")
    else:
        st.markdown(f"共检索到 **{len(retrieval_results)}** 条相关信息")
        
        for i, result in enumerate(retrieval_results[:5]):  # 只显示前5条
            score = result.get("score", 0)
            title = result.get("title", f"结果 {i+1}")
            text = result.get("text", "")[:200]
            
            with st.expander(f"📄 {title} (相似度: {score:.3f})"):
                st.text(text + "..." if len(result.get("text", "")) > 200 else text)
    
    st.divider()
    
    # 4. 自我反思
    st.markdown("#### 🔎 自我反思")
    
    reflection = current_state.get("self_reflection")
    
    if reflection:
        quality_score = reflection.get("quality_score", 0)
        needs_improvement = reflection.get("needs_improvement", False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("质量评分", f"{quality_score:.2f}")
        with col2:
            if needs_improvement:
                st.warning("⚠️ 需要改进")
            else:
                st.success("✅ 质量合格")
        
        suggestions = reflection.get("suggestions", [])
        if suggestions:
            st.markdown("**改进建议**:")
            for s in suggestions:
                st.markdown(f"- {s}")
    else:
        st.info("未启用自我反思或反思结果暂不可用")
    
    st.divider()
    
    # 5. 状态指示
    st.markdown("#### 📌 执行状态")
    
    status = current_state.get("status", "idle")
    status_config = {
        "idle": ("⭕ 空闲", "status-running"),
        "running": ("⏳ 运行中", "status-running"),
        "waiting_human": ("👤 等待人工确认", "status-waiting"),
        "completed": ("✅ 完成", "status-completed"),
        "error": ("❌ 错误", "status-error")
    }
    
    status_text, status_class = status_config.get(status, ("❓ 未知", "status-running"))
    st.markdown(f'<span class="status-badge {status_class}">{status_text}</span>', unsafe_allow_html=True)


def render_chat_area():
    """渲染聊天区域"""
    st.markdown("### 💬 开启对话分析")
    
    # 聊天历史容器
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            # 用户消息 - 右侧气泡
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin: 12px 0;">
                <div style="
                    background: #d8b4fe;
                    color: white;
                    border-radius: 20px 20px 4px 20px;
                    padding: 12px 18px;
                    max-width: 70%;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                ">
                    <div style="font-size: 1.8em; margin-bottom: 5px; opacity: 0.8;">👤 Client</div>
                    <div>{chat['question']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Agent 回复 - 左侧气泡
            if 'answer' in chat:
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 12px 0;">
                    <div style="
                        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
                        color: #1a1a1a;
                        border-radius: 20px 20px 20px 4px;
                        padding: 12px 18px;
                        max-width: 70%;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                    ">
                        <div style="font-size: 1.8em; margin-bottom: 5px; color: #667eea; font-weight: bold;">🤖 Agent</div>
                        <div>{chat['answer']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    # 输入区域
    st.markdown("#### ✏️ 输入问题")
    
    user_input = st.text_area(
        "请输入您的核电材料问题",
        placeholder="例如：ODS钢的高温性能评价有哪些相关标准？",
        height=100,
        key="user_input_area"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_button = st.button("🚀 开始分析", type="primary", use_container_width=True)
    with col2:
        if st.button("📋 快捷问题", use_container_width=True):
            st.info("快捷问题模板开发中...")
    
    # 处理提交
    if submit_button and user_input:
        run_analysis(user_input)


def run_analysis(user_input: str):
    """运行 Agent 分析"""
    
    # 显示进度
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. 初始化
        status_text.markdown("🔄 初始化 Agent...")
        progress_bar.progress(10)
        time.sleep(0.3)
        
        # 2. 任务理解
        status_text.markdown("🧠 理解问题...")
        progress_bar.progress(20)
        
        # 创建初始状态
        initial_state = create_initial_state(user_input, st.session_state.session_id)
        
        # 3. 运行 Agent
        status_text.markdown("🔄 执行 Agent 流程...")
        progress_bar.progress(40)
        
        # 获取 Agent 图
        agent = get_agent_graph()
        
        # 运行并逐步更新 UI
        with st.spinner("Agent 正在思考中..."):
            final_state = agent.invoke(initial_state)
        
        progress_bar.progress(70)
        status_text.markdown("✅ 分析完成，整理结果...")
        
        # 保存到 session_state
        st.session_state.current_agent_state = final_state
        
        # 4. 获取答案
        final_answer = final_state.get("final_answer", "分析完成，但未生成答案")
        
        progress_bar.progress(90)
        
        # 5. 添加到历史记录
        chat_entry = {
            "time": datetime.now().isoformat(),
            "question": user_input,
            "answer": final_answer,
            "state": final_state,
            "task_type": final_state.get("current_task", "unknown")
        }
        st.session_state.chat_history.append(chat_entry)
        
        progress_bar.progress(100)
        status_text.markdown("✅ 完成!")
        
        # 重新运行页面以显示结果
        st.rerun()
        
    except Exception as e:
        progress_bar.progress(0)
        st.error(f"分析过程中出错: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# ============================================
# 主程序
# ============================================
def main():
    """主程序入口"""
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
