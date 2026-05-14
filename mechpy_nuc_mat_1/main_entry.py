"""
核电材料智能分析平台 - 主入口（美化版）
Nuclear Materials AI Analysis Platform
"""
import streamlit as st
import sys
import os

# 页面配置
st.set_page_config(
    page_title="核电材料辅助分析系统",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式（现代化玻璃质感主题 + 超大标题 + 对称布局）
st.markdown("""
<style>
    /* 引入 Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Noto+Sans+SC:wght@400;500;700;900&display=swap');

    /* CSS 变量 */
    :root {
        --primary: #3b82f6;
        --primary-dark: #1e40af;
        --secondary: #06b6d4;
        --bg-white: #ffffff;
        --card-bg: #ffffff;
        --card-border: #e2e8f0;
        --text-dark: #1e293b;
        --text-muted: #64748b;
        --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.02);
        --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
        --radius-xl: 24px;
        --radius-lg: 16px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* 全局字体 */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans SC', sans-serif;
    }

    /* 白色背景 */
    .stApp {
        background: #ffffff;
    }

    /* 自定义滚动条 */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f8fafc;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary), var(--secondary));
        border-radius: 10px;
    }


    /* 主标题 - 超大版本 */
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: 900;
        background: linear-gradient(135deg, #0f172a 0%, #3b82f6 40%, #06b6d4 70%, #0ea5e9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: 3px;
        animation: fadeInDown 0.8s ease-out;
        line-height: 1.1;
        text-shadow: 0 2px 10px rgba(59, 130, 246, 0.2);
    }

    /* 副标题 */
    .subtitle {
        text-align: center;
        color: var(--text-muted);
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 3rem;
        letter-spacing: 2px;
        animation: fadeInUp 0.8s ease-out;
    }

    /* 动画定义 */
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(59, 130, 246, 0.3); }
        50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.6); }
        100% { box-shadow: 0 0 5px rgba(59, 130, 246, 0.3); }
    }

    /* 功能卡片容器 - 完全居中对称 */
    .card-container {
        display: flex;
        justify-content: center;
        align-items: stretch;
        gap: 3rem;
        padding: 1rem 0 2rem;
        max-width: 1300px;
        margin: 0 auto;
    }

    /* 功能卡片 - 玻璃质感 */
    .feature-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: var(--radius-xl);
        padding: 3rem 2.5rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--glass-border);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.8s ease-out;
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        min-width: 0;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.4) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }

    .feature-card:hover::before {
        opacity: 1;
    }

    .feature-card:hover {
        transform: translateY(-12px);
        box-shadow: var(--shadow-lg);
        border-color: rgba(59, 130, 246, 0.3);
    }

    /* 图标 */
    .card-icon {
        font-size: 5rem;
        margin-bottom: 1.5rem;
        display: block;
        text-align: center;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.15));
    }

    /* 卡片标题 */
    .card-title {
        color: var(--text-dark);
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        letter-spacing: -0.5px;
    }

    /* 卡片描述 */
    .card-desc {
        color: var(--text-muted);
        font-size: 1rem;
        line-height: 1.8;
        text-align: center;
        margin-bottom: 2rem;
        flex: 1;
        display: flex;
        align-items: center;
    }

    /* 标签 */
    .card-tags {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }

    .tag {
        background: rgba(59, 130, 246, 0.1);
        color: #1e40af;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 500;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(59, 130, 246, 0.15);
        transition: var(--transition);
    }

    .tag:hover {
        background: rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.3);
        transform: scale(1.05);
    }

    /* 自定义 Streamlit 按钮 - 替代原版 */
    .stButton > button {
        width: 100% !important;
        padding: 14px 24px !important;
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        margin-top: 0.5rem;
        cursor: pointer;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
        background: linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%) !important;
        animation: glow 1.5s infinite;
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(59, 130, 246, 0.4);
    }

    /* 返回首页按钮特殊样式 */
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0,0,0,0.05) !important;
        color: var(--text-dark) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: white !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* 底部信息 */
    .footer-info {
        text-align: center;
        padding: 2rem 1rem;
        color: #94a3b8;
        font-size: 0.9rem;
        border-top: 1px solid rgba(0,0,0,0.05);
        margin-top: 1rem;
    }

    .footer-info a {
        color: var(--primary);
        text-decoration: none;
        font-weight: 500;
    }

    .divider {
        display: flex;
        align-items: center;
        text-align: center;
        margin: 2.5rem 0 1.5rem;
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
    }

    .divider::before,
    .divider::after {
        content: '';
        flex: 1;
        border-bottom: 2px solid rgba(0,0,0,0.06);
    }

    .divider span {
        padding: 0 1.5rem;
        background: transparent;
    }

    /* 响应式调整 */
    @media (max-width: 1200px) {
        .main-title {
            font-size: 4rem;
        }
        .card-container {
            gap: 2rem;
            padding: 1rem 2rem;
        }
        .feature-card {
            padding: 2.5rem 2rem;
        }
    }

    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        .card-container {
            flex-direction: column;
            padding: 1rem;
        }
        .feature-card {
            width: 100%;
            padding: 2rem 1.5rem;
        }
        .card-icon {
            font-size: 3.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


def show_homepage():
    """显示首页（美化版 - 对称布局）"""
    # 页面头部
    st.markdown('<h1 class="main-title" style="font-size: 36px !important;">核电材料辅助分析系统</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p class="subtitle">
        Nuclear Materials AI Analysis Platform · 集成 RAG 与 Agent 的智能分析平台
    </p>
    """, unsafe_allow_html=True)

    # 功能卡片区域 - 完全对称的两列布局
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # RAG智能问答卡片
        st.markdown("""
        <div class="feature-card">
            <span class="card-icon">📚</span>
            <span class="card-title">智能问答</span>
            <p class="card-desc">
                基于 RAG（检索增强生成）技术，从知识库中检索相关文献，
                结合大语言模型进行专业分析回答。支持多模态检索和知识图谱集成。
            </p>
            <div class="card-tags">
                <span class="tag">RAG</span>
                <span class="tag">知识图谱</span>
                <span class="tag">文献检索</span>
                <span class="tag">向量检索</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🎶 进入智能问答", key="rag_page", use_container_width=True):
            st.session_state.current_page = "rag"
            st.rerun()

    with col_right:
        # Agent助手卡片 - 完全对称
        st.markdown("""
        <div class="feature-card">
            <span class="card-icon">🤖</span>
            <span class="card-title">Agent 助手</span>
            <p class="card-desc">
                基于 LangGraph 构建的智能 Agent，支持多步推理、自我反思、
                人在回路审核。可进行材料对比、标准匹配、性能评估等专业任务。
            </p>
            <div class="card-tags">
                <span class="tag">Agent</span>
                <span class="tag">多轮推理</span>
                <span class="tag">人机协同</span>
                <span class="tag">自我反思</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✨ 进入 Agent 助手", key="agent_page", use_container_width=True):
            st.session_state.current_page = "agent"
            st.rerun()

    # 底部信息
    st.markdown("""
    <div class="divider">
        <span>⚡ 核安全提示：分析结果仅供研究参考</span>
    </div>
    <div class="footer-info">
        <p>© 核电材料智能分析平台 · zyx377987701@163.com </p>
    </div>
    """, unsafe_allow_html=True)


def load_rag_page():
    """加载RAG智能问答页面"""
    try:
        from api.nuc_web_server import setup_web_server
        setup_web_server()
    except ImportError as e:
        st.error(f"无法加载智能问答模块: {e}")
        if st.button("🔙 首页"):
            st.session_state.current_page = "home"
            st.rerun()


def load_agent_page():
    """加载Agent助手页面"""
    try:
        from nuc_mat_agent.web_ui import main as agent_main
        agent_main()
    except ImportError as e:
        st.error(f"无法加载Agent模块: {e}")
        if st.button("🔙 首页"):
            st.session_state.current_page = "home"
            st.rerun()


def main():
    """主函数"""
    # 初始化session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"

    # 非首页时显示一个美观的返回按钮
    if st.session_state.current_page != "home":
        cols = st.columns([1, 6, 1])
        with cols[0]:
            if st.button("🏠 首页", key="return_home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
        st.markdown("---")

    # 根据当前页面显示相应内容
    if st.session_state.current_page == "home":
        show_homepage()
    elif st.session_state.current_page == "rag":
        load_rag_page()
    elif st.session_state.current_page == "agent":
        load_agent_page()


if __name__ == "__main__":
    main()