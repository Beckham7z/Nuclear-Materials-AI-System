# v8版本更新日志：rag能够使用微调模型进行分析，完善前端展示界面

# 核心模块
import sys
import os
import yaml
import platform
import time
import uuid
import re
import json
from pyvis.network import Network
from datetime import datetime
from PIL import Image
import streamlit as st
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# 项目配置模块
from configuration.logset import logger
from configuration.unit_config import LLMConfig, DatabaseConfig, Neo4jConfig, MongoConfig, MilvusConfig, RAGConfig
from configuration.global_config import GlobalConfig

# 查询分发
from api.image_first import ImageQueryHandler # 图像查询处理器
from api.absdract_first import AbstractQueryEngine # 摘要查询处理器
from api.node_first import LocalQueryEngine # 知识图谱查询处理器
from api.ReserchPlan import ReserchPlanHandler # 研究计划生成处理器
from api.PerformanceAnalysis import PerformanceAnalysisHandler # 性能分析处理器

# 异步运行
from llm.async_utils import run_async
import asyncio

# LightRAG 相关导入
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
# from lightrag.llm import hfp_model_complete
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# transformers 用于直接加载模型
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 可视化模块导入
from visualization.lightrag_visualizer import LightRAGVisualizer
from visualization.enhanced_lightrag_visualizer import EnhancedLightRAGVisualizer
# mongo检索
from mongo_response.mongo_utils import search_mongo_documents, build_enhanced_prompt, handle_mongo_query

"""
前端界面构建 (setup_web_server函数前半部分，约70-380行)
模型检索与处理逻辑 (setup_web_server函数中间部分，约380-550行)
结果反馈与可视化 (setup_web_server函数后半部分，约550-800行)
辅助功能函数 (Lines 800-1150)

V7版本新增：
- 左侧侧边栏添加微调模型选择器
- 支持加载PEFT微调后的LoRA适配器
- 可在微调模型和基础模型之间切换
"""

def setup_web_server():
    # 初始化处理过程日志
    if "process_logs" not in st.session_state:
        st.session_state.process_logs = []
    
    # 获取项目根目录的相对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(project_root, "api", "config.yaml")
    # 支持的机构列表：
    ins_list = ["deepseek", "ollama", "openai", "zhipu", "siliconflow", "flagembed", "xinference"]

    # 设置页面
    st.set_page_config(
        page_title="Nuclear Material Analysis Interface with RAG",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        /* 进度条样式 */
        .progress-container {
            background: #f0f2f6; 
            padding: 15px; 
            border-radius: 10px; 
            margin: 10px 0;
            border: 1px solid #e0e0e0;
        }
        .progress-header {
            display: flex; 
            justify-content: space-between; 
            margin-bottom: 8px;
            font-weight: bold;
        }
        .progress-bar {
            background: #e0e0e0; 
            border-radius: 10px; 
            height: 10px;
            overflow: hidden;
        }
        .progress-fill {
            background: linear-gradient(90deg, #3c6382, #1e3799); 
            height: 10px; 
            border-radius: 10px; 
            transition: width 0.5s ease;
        }
        .progress-subtext {
            margin-top: 8px; 
            font-size: 0.85em; 
            color: #666;
            font-style: italic;
        }
        /* 加载动画 */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3c6382;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        /* RAG 特定样式 */
        .RAG-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        /* 微调模型样式 */
        .fine-tuned-badge {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    # 自定义侧边栏（保持不变）
    with st.sidebar:
        # 1. 读取yaml文件按钮和输入框
        st.markdown("## Nuclear Material Analysis Platform")
        st.markdown("### 配置文件管理")
        default_yaml_path = yaml_path
        yaml_input_path = st.text_input(
            "YAML 文件路径", value=default_yaml_path, key="yaml_input_path")
        if st.button("读取YAML配置", key="load_yaml_btn"):
            try:
                with open(yaml_input_path, 'r', encoding='utf-8') as f:
                    # 将配置保存到 session_state
                    st.session_state["config"] = yaml.safe_load(f)
                st.success("YAML 配置读取成功！")
            except Exception as e:
                st.error(f"读取失败: {e}")

        # 确保配置存在
        if "config" not in st.session_state:
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    st.session_state["config"] = yaml.safe_load(f)
            except Exception as e:
                st.error(f"初始化配置失败: {e}")
                st.session_state["config"] = {}  # 使用空配置作为后备

        config = st.session_state["config"]  # 获取当前配置

        # =====================================================
        # V7版本新增：微调模型配置区域
        # =====================================================
        st.markdown("## 🔧 微调模型配置")
        with st.expander("微调模型", expanded=True):
            # 微调模型路径（默认值使用相对路径）
            default_fine_tuned_path = os.path.join(project_root, "src", "fine_tuning", "output", "nuclear_sft_dapt_swift_fast", "v0-20260310-114209", "checkpoint-1890")
            fine_tuned_model_path = st.text_input(
                "微调模型路径",
                value=default_fine_tuned_path,
                key="fine_tuned_model_path"
            )
            
            # 基础模型路径
            # base_model_path = st.text_input(
            #     "基础模型路径",
            #     value="/home/zyx/.cache/modelscope/hub/models/Qwen/Qwen3.5-2B",
            #     key="base_model_path"
            # )
            
            # 是否启用微调模型
            use_fine_tuned = st.toggle(
                "启用微调模型",
                value=True,
                key="use_fine_tuned",
                help="开启后使用微调后的LoRA适配器，否则使用基础模型"
            )
            
            if use_fine_tuned:
                st.markdown('<span class="fine-tuned-badge">✓ 已启用微调模型</span>', unsafe_allow_html=True)
                st.info(f"微调模型: {os.path.basename(fine_tuned_model_path)}")
            else:
                st.warning("当前使用基础模型")
        
        # =====================================================
        
        # 材料数据库选择
        st.markdown("## 材料数据库")
        material_db = st.selectbox(
            "选择材料数据库",
            ["核电材料性能数据库", "辐射效应数据库", "腐蚀数据中心", "热老化数据库", "自定义数据库"],
            key="material_db_select"
        )

        # 核电材料类型过滤
        st.markdown("## 材料类型过滤")
        with st.expander("材料类型", expanded=True):
            st.multiselect(
                "选择材料类别",
                ["压力容器钢", "燃料包壳材料", "结构合金", "高温合金", "复合材料", "混凝土", "密封材料"],
                default=["压力容器钢", "燃料包壳材料"],
                key="material_type_select"
            )

            st.multiselect(
                "选择关注性能",
                ["抗辐射性", "高温强度", "耐腐蚀性", "疲劳寿命", "热导率", "蠕变特性", "抗氧化性"],
                default=["抗辐射性", "耐腐蚀性"],
                key="material_property_select"
            )

        # Chat LLM
        st.markdown("## 分析模型配置")
        with st.expander("核电材料分析模型", expanded=False):
            # 将默认索引改为 deepseek
            default_index = ins_list.index("deepseek") if "deepseek" in ins_list else 0
            institution_chat = st.selectbox(
                "选择AI模型提供商", ins_list, index=default_index, key="institution_chat_select")
            if institution_chat.lower() in config.get('llm', {}):
                inst_config = config['llm'][institution_chat.lower()]
                st.text_input(
                    "模型名称",
                    value=inst_config.get('model', ''),
                    key="model_chat_input"
                )
                st.text_input(
                    "API 密钥",
                    value=inst_config.get('api_key', ''),
                    type="password",
                    key="api_key_chat_input"
                )
                st.text_input(
                    "服务器地址",
                    value=inst_config.get('base_url', ''),
                    key="base_url_chat_input"
                )
            else:
                st.warning(f"未找到 {institution_chat} 的配置信息")
                st.text_input("模型名称", value="", key="model_chat_input")
                st.text_input("API 密钥", value="", type="password",
                              key="api_key_chat_input")
                st.text_input("服务器地址", value="", key="base_url_chat_input")

            st.slider(
                "输出随机性 (Temperature)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,  # 核电材料分析需要更高的确定性，降低随机性
                step=0.1,
                key="temperature_chat_slider"
            )
            st.slider(
                "最大输出长度 (Tokens)",
                min_value=128,
                max_value=8192,
                value=4096,
                step=16,
                key="max_tokens_chat_slider"
            ) 
        
               # 辅助 LLM 配置（Rerank、VL、Embedding 合并）
        st.markdown("## 🛠️ 辅助 LLM")
        with st.expander("辅助 LLM 配置", expanded=False):
            st.caption("包含 Rerank、VL、Embedding 模型的统一配置区域")
            
            # Embedding LLM
            st.markdown("### Embedding")
            default_embed_index = ins_list.index("ollama") if "ollama" in ins_list else 0
            institution_embed = st.selectbox(
                "Institution",
                ins_list,
                index=default_embed_index,
                key="institution_embed_select"
            )
            if institution_embed.lower() in config.get('llm', {}):
                inst_config = config['llm'][institution_embed.lower()]
                model_embed = st.text_input(
                    "Model Name",
                    value=inst_config.get('embed_model', ''),
                    key="model_embed_input"
                )
                api_key_embed = st.text_input(
                    "API Key",
                    value=inst_config.get('api_key', ''),
                    type="password",
                    key="api_key_embed_input"
                )
                base_url_embed = st.text_input(
                    "Base URL",
                    value=inst_config.get('base_url', ''),
                    key="base_url_embed_input"
                )
            
            st.markdown("---")
            
            # Rerank LLM
            st.markdown("### Rerank")
            default_rerank_index = ins_list.index(
                "siliconflow") if "siliconflow" in ins_list else 0
            institution_rerank = st.selectbox(
                "Institution",
                ins_list,
                index=default_rerank_index,
                key="institution_rerank_select"
            )
            if institution_rerank.lower() in config.get('llm', {}):
                inst_config = config['llm'][institution_rerank.lower()]
                model_rerank = st.text_input(
                    "Model Name",
                    value=inst_config.get('rerank_model', ''),
                    key="model_rerank_input"
                )
                api_key_rerank = st.text_input(
                    "API Key",
                    value=inst_config.get('api_key', ''),
                    type="password",
                    key="api_key_rerank_input"
                )
                base_url_rerank = st.text_input(
                    "Base URL",
                    value=inst_config.get('base_url', ''),
                    key="base_url_rerank_input"
                )
            
            st.markdown("---")
            
            # VL LLM
            st.markdown("### VL")
            default_vl_index = ins_list.index(
                "deepseek") if "deepseek" in ins_list else 0
            institution_vl = st.selectbox(
                "Institution",
                ins_list,
                index=default_vl_index,
                key="institution_vl_select"
            )
            if institution_vl.lower() in config.get('llm', {}):
                inst_config = config['llm'][institution_vl.lower()]
                model_vl = st.text_input(
                    "Model Name",
                    value=inst_config.get('vl_model', ''),
                    key="model_vl_input"
                )
                api_key_vl = st.text_input(
                    "API Key",
                    value=inst_config.get('api_key', ''),
                    type="password",
                    key="api_key_vl_input"
                )
                base_url_vl = st.text_input(
                    "Base URL",
                    value=inst_config.get('base_url', ''),
                    key="base_url_vl_input"
                )
                temperature_vl = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    key="temperature_vl_slider"
                )
                max_tokens_vl = st.slider(
                    "Max Tokens",
                    min_value=128,
                    max_value=8192,
                    value=4096,
                    step=16,
                    key="max_tokens_vl_slider"
                )

        # # Embedding LLM
        # st.markdown("## Embedding")
        # with st.expander("Embedding LLM", expanded=False):
        #     default_embed_index = ins_list.index("ollama") if "ollama" in ins_list else 0
        #     institution_embed = st.selectbox(
        #         "Institution",
        #         ins_list,
        #         index=default_embed_index,
        #         key="institution_embed_select"
        #     )
        #     if institution_embed.lower() in config.get('llm', {}):
        #         inst_config = config['llm'][institution_embed.lower()]
        #         model_embed = st.text_input(
        #             "Model Name",
        #             value=inst_config.get('embed_model', ''),
        #             key="model_embed_input"
        #         )
        #         api_key_embed = st.text_input(
        #             "API Key",
        #             value=inst_config.get('api_key', ''),
        #             type="password",
        #             key="api_key_embed_input"
        #         )
        #         base_url_embed = st.text_input(
        #             "Base URL",
        #             value=inst_config.get('base_url', ''),
        #             key="base_url_embed_input"
        #         )
        
                    # RAG 配置
        st.markdown("## RAG 配置")
        with st.expander("RAG 设置", expanded=False):
            st.info("RAG 是一个高效的检索增强生成框架，支持多模态检索和知识图谱集成")
            
            # RAG工作目录（默认值使用相对路径）
            default_rag_dir = os.path.join(project_root, "myKG")
            RAG_working_dir = st.text_input(
                "工作目录", 
                value=default_rag_dir,
                key="RAG_working_dir"
            )
            
            # 根据是否启用微调模型显示不同的LLM选项
            if use_fine_tuned:
                RAG_llm_model = st.text_input(
                    "LLM 模型名称",
                    value="qwen2.5:1.5b-instruct-q4_K_S",  # 微调模型标识
                    key="RAG_llm_model"
                )
                st.success("✓ 使用微调模型进行RAG生成")
            else:
                RAG_llm_model = st.text_input(
                    "LLM 模型名称",
                    value="qwen2.5:1.5b-instruct-q4_K_S",  # 使用小模型提高速度
                    key="RAG_llm_model"
                )
            
            RAG_embed_model = st.text_input(
                "Embedding 模型",
                value="bge-m3:latest",
                key="RAG_embed_model"
            )
            
            RAG_ollama_host = st.text_input(
                "Ollama 服务器地址",
                value="http://127.0.0.1:11434",
                key="RAG_ollama_host"
            )

        # # Rerank LLM
        # st.markdown("## Rerank")
        # with st.expander("Rerank LLM", expanded=False):
        #     default_rerank_index = ins_list.index(
        #         "siliconflow") if "siliconflow" in ins_list else 0
        #     institution_rerank = st.selectbox(
        #         "Institution",
        #         ins_list,
        #         index=default_rerank_index,
        #         key="institution_rerank_select"
        #     )
        #     if institution_rerank.lower() in config.get('llm', {}):
        #         inst_config = config['llm'][institution_rerank.lower()]
        #         model_rerank = st.text_input(
        #             "Model Name",
        #             value=inst_config.get('rerank_model', ''),
        #             key="model_rerank_input"
        #         )
        #         api_key_rerank = st.text_input(
        #             "API Key",
        #             value=inst_config.get('api_key', ''),
        #             type="password",
        #             key="api_key_rerank_input"
        #         )
        #         base_url_rerank = st.text_input(
        #             "Base URL",
        #             value=inst_config.get('base_url', ''),
        #             key="base_url_rerank_input"
        #         )

        # # VL LLM
        # st.markdown("## VL")
        # with st.expander("VL LLM", expanded=False):
        #     default_vl_index = ins_list.index(
        #         "deepseek") if "deepseek" in ins_list else 0
        #     institution_vl = st.selectbox(
        #         "Institution",
        #         ins_list,
        #         index=default_vl_index,
        #         key="institution_vl_select"
        #     )
        #     if institution_vl.lower() in config.get('llm', {}):
        #         inst_config = config['llm'][institution_vl.lower()]
        #         model_vl = st.text_input(
        #             "Model Name",
        #             value=inst_config.get('vl_model', ''),
        #             key="model_vl_input"
        #         )
        #         api_key_vl = st.text_input(
        #             "API Key",
        #             value=inst_config.get('api_key', ''),
        #             type="password",
        #             key="api_key_vl_input"
        #         )
        #         base_url_vl = st.text_input(
        #             "Base URL",
        #             value=inst_config.get('base_url', ''),
        #             key="base_url_vl_input"
        #         )
        #         temperature_vl = st.slider(
        #             "Temperature",
        #             min_value=0.0,
        #             max_value=1.0,
        #             value=0.7,
        #             step=0.1,
        #             key="temperature_vl_slider"
        #         )
        #         max_tokens_vl = st.slider(
        #             "Max Tokens",
        #             min_value=128,
        #             max_value=8192,
        #             value=4096,
        #             step=16,
        #             key="max_tokens_vl_slider"
        #         )

 
        # 数据库配置
        st.markdown("## 数据库配置")

        # Neo4j - 核电材料知识图谱
        with st.expander("核电材料知识图谱 (Neo4j)", expanded=False):
            neo4j_url = st.text_input(
                "URL", value=config.get('neo4j', {}).get('url', ''), key="neo4j_url")
            neo4j_username = st.text_input(
                "用户名", value=config.get('neo4j', {}).get('username', ''), key="neo4j_username")
            neo4j_password = st.text_input(
                "密码", value=config.get('neo4j', {}).get('password', ''), type="password", key="neo4j_password")
            neo4j_database = st.text_input(
                "数据库名", value=config.get('neo4j', {}).get('database', ''), key="neo4j_database")

        # MongoDB - 材料文档数据库
        with st.expander("材料文档数据库 (MongoDB)", expanded=False):
            mongo_url = st.text_input(
                "URL", value=config.get('mongo', {}).get('url', ''), key="mongo_url")
            mongo_database = st.text_input(
                "数据库名", value=config.get('mongo', {}).get('database', ''), key="mongo_database")

        # Milvus - 材料性能向量数据库
        with st.expander("材料性能向量数据库 (Milvus)", expanded=False):
            milvus_url = st.text_input(
                "URL", value=config.get('milvus', {}).get('url', ''), key="milvus_url")
            milvus_username = st.text_input(
                "用户名", value=config.get('milvus', {}).get('username', ''), key="milvus_username")
            milvus_password = st.text_input(
                "密码", value=config.get('milvus', {}).get('password', ''), type="password", key="milvus_password")
            milvus_database = st.text_input(
                "数据库名", value=config.get('milvus', {}).get('database', ''), key="milvus_database")


        # 导出yaml按钮
        if platform.system() == "Windows":
            default_export_path = os.path.expanduser("~\\nuclear_material_config.yaml")
        elif platform.system() == "Darwin":
            default_export_path = os.path.expanduser("~/nuclear_material_config.yaml")
        else:
            default_export_path = os.path.expanduser("~/nuclear_material_config.yaml")

        export_path = st.text_input(
            "导出配置路径", value=default_export_path, key="export_yaml_path")
        if st.button("导出当前配置", key="export_yaml_btn"):
            try:
                with open(export_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, allow_unicode=True)
                st.success(f"已导出到: {export_path}")
            except Exception as e:
                st.error(f"导出失败: {e}")

    # 主页面内容
    st.markdown('# ⚛️ 核电材料智能分析平台 ', unsafe_allow_html=True)

    # RAG 介绍
    st.markdown("""
    <div class="RAG-section">
        <h3>🚀 增强功能</h3>
        <p>本平台集成了 RAG 框架，提供更高效的检索增强生成能力：</p>
        <ul>
            <li>多模态文档检索</li>
            <li>知识图谱集成</li>
            <li>智能语义搜索</li>
            <li>实时知识更新</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 显示当前模型状态
    if use_fine_tuned:
        st.markdown("""
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #f8f9fa; 
                    color: #666; padding: 10px 15px; border-radius: 8px; 
                    margin: 10px 0; text-align: center;">
            <strong>📌 当前使用：基础模型</strong>
        </div>
        """, unsafe_allow_html=True)

    # # 安全提示
    # st.markdown("""
    # <div class="nuclear-warning">
    #     <strong>注意：</strong>本平台提供的分析结果仅供研究参考，实际核电材料应用需遵循相关安全标准和规范。
    # </div>
    # """, unsafe_allow_html=True)

    # 参数设置部分
    st.markdown('## 分析参数配置', unsafe_allow_html=True)
    st.markdown('<hr style="margin: 0.5rem 0 1rem 0; border-width: 3px; border-color: #3c6382;">',
                unsafe_allow_html=True)

    col_query, col_rag = st.columns([1, 1])
    with col_query:
        # 查询方式选择 - 核电材料专用
        query_method = st.radio(
            "请选择分析方法:",
            ("RAG智能检索",  # 别修改 名称锁死容易错
             "辐射效应评估",
             "腐蚀行为预测",
             "寿命评估与预测",
             "材料替代方案研究",
             "失效模式分析",
             "材料性能分析"),  # 新增 RAG 选项
            key="query_method",
            help="选择适合您问题的分析方法"
        )

    with col_rag:
        # 核电材料分析专用参数
        st.checkbox(
            "考虑辐照损伤累积效应",
            key="radiation_damage",
            help="分析中纳入长期辐照损伤累积对材料性能的影响"
        )

        st.checkbox(
            "启用安全系数校准",
            key="safety_factor",
            help="根据核电规范自动校准安全系数"
        )

        # Top K 滑块，设置检索返回的结果数量上限
        top_k_value = st.slider(
            "参考文献数量",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            key="top_k",
            help="设置分析时参考的文献和数据数量"
        )

    st.markdown('## 问题输入', unsafe_allow_html=True)
    st.markdown('<hr style="margin: 0.5rem 0 1rem 0; border-width: 3px; border-color: #3c6382;">',
                unsafe_allow_html=True)

    col_input, col_button = st.columns([4, 1])

    with col_input:
        user_message = st.text_area(
            "请输入您的核电材料问题",
            value="基于你的知识库，分析目前哪些材料适用于核反应堆设计？",
            key="user_message_area",
            height=150
        )

        # 附加参数 - 问题相关条件
        with st.expander("问题附加条件", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.selectbox(
                    "反应堆类型",
                    ["压水堆(PWR)", "沸水堆(BWR)", "重水堆(HWR)", "快中子堆(FBR)", "其他"],
                    key="reactor_type"
                )
            with col2:
                st.text_input(
                    "工作温度 (°C)",
                    value="300-350",
                    key="operating_temperature"
                )
            with col3:
                st.text_input(
                    "预期寿命 (年)",
                    value="40",
                    key="expected_lifetime"
                )

    with col_button:
        st.markdown(
            """
            <style>
            div[data-testid="column"] button {
                height: 150px !important;
                width: 100%;
                font-size: 16px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        btn_clicked = st.button("获取分析结果", key="get_response_btn")

    # 核电材料分析专用提示词
    prompt = f"""你是核电材料领域的专家，拥有深厚的材料科学和核工程知识。
    请基于提供的问题和参考数据，提供专业、准确、客观的分析结果。
    分析应包括：
    1. 基于现有研究和数据的科学结论
    2. 不同材料选项的优缺点对比
    3. 实际应用中的注意事项和限制条件
    4. 相关安全标准和规范的参考
    5. 建议的进一步研究方向（如适用）

    请使用专业术语，但保持表述清晰易懂，避免过度简化可能导致的误解。
    对于不确定的信息，应明确说明并提供可能的误差范围。
    """

    st.markdown('## 分析过程与结果', unsafe_allow_html=True)
    st.markdown('<hr style="margin: 0.5rem 0 1rem 0; border-width: 3px; border-color: #3c6382;">',
                unsafe_allow_html=True)

    # 添加处理过程展示区域
    process_container = st.container()
    with process_container:
        st.subheader("处理进度")
        process_placeholder = st.empty()

    # 结果展示区域
    result_container = st.container()

    # 当按钮被点击，才执行下述操作
    if btn_clicked:
        # 重置处理日志
        st.session_state.process_logs = []
        st.session_state.cancel_requested = False  # 添加这行
        
        with result_container:
            st.subheader("分析结果")
            result_placeholder = st.empty()

        # 创建进度条占位符
        progress_placeholder = st.empty()
        
        try:
            # 定义详细的处理步骤（使用原来的中文提示）
            processing_steps = [
                {"main": "开始处理用户请求", "module": "系统", "duration": 0.5},
                {"main": f"解析用户问题: {user_message[:60]}...", "module": "自然语言处理模块", "duration": 1.0},
                {"main": "验证分析配置参数", "module": "配置管理模块", "duration": 0.5},
                {"main": "构建分析所需的全局配置", "module": "系统核心模块", "duration": 1.0},
                {"main": "从材料文档数据库检索相关信息", "module": "MongoDB检索模块", "duration": 1.2},
                {"main": "生成专业分析提示词", "module": "提示词工程模块", "duration": 0.6},
                {"main": f"调用{st.session_state.get('institution_chat_select')}模型进行分析", "module": "AI服务接口", "duration": 2.0},
                {"main": "正在获取AI模型分析结果", "module": "AI服务接口", "duration": 2.5},
                {"main": "分析完成，整理结果", "module": "结果处理模块", "duration": 0.7}
            ]
            
            # 如果是 RAG 查询，修改处理步骤
            if query_method == "RAG智能检索":
                use_ft = st.session_state.get("use_fine_tuned", False)
                model_desc = "微调模型" if use_ft else "基础模型"
                processing_steps = [
                    {"main": "开始处理用户请求", "module": "系统", "duration": 0.5},
                    {"main": f"解析用户问题: {user_message[:60]}...", "module": "自然语言处理模块", "duration": 1.0},
                    {"main": "验证RAG配置参数", "module": "RAG配置模块", "duration": 0.5},
                    {"main": f"初始化RAG引擎 (使用{model_desc})", "module": "RAG核心模块", "duration": 1.5},
                    {"main": "执行多模态文档检索", "module": "RAG检索模块", "duration": 2.0},
                    {"main": "构建知识图谱查询", "module": "知识图谱模块", "duration": 1.0},
                    {"main": "生成增强提示词", "module": "提示词工程模块", "duration": 0.6},
                    {"main": f"调用{model_desc}进行综合分析", "module": "AI服务接口", "duration": 2.5},
                    {"main": "整理RAG分析结果", "module": "结果处理模块", "duration": 0.7}
                ]
            
            # 执行每个步骤
            for i, step in enumerate(processing_steps):
                # 显示进度条
                show_advanced_progress(
                    progress_placeholder, 
                    i + 1, 
                    len(processing_steps),
                    f"[{step['module']}] {step['main']}",
                    f"步骤 {i+1}/{len(processing_steps)} - 正在处理中..."
                )
                
                # 记录处理日志（使用原来的模块名称和提示）
                log_process(step["main"], step["module"], process_placeholder)
                
                # 根据步骤执行实际处理逻辑
                if i == 0:  # 开始处理用户请求
                    pass  # 无需额外操作
                    
                elif i == 1:  # 解析用户问题
                    # 这里可以添加问题解析逻辑
                    pass
                    
                elif i == 2:  # 验证配置参数
                    # 验证配置的完整性
                    if not st.session_state.get("model_chat_input"):
                        raise ValueError("请配置AI模型参数")
                        
                elif i == 3:  # 构建分析配置
                    global_config = GlobalConfig(
                        chat=LLMConfig(
                            model_type="chat",
                            institution=st.session_state.get("institution_chat_select"),
                            model=st.session_state.get("model_chat_input"),
                            prompt=prompt,
                            user_message=user_message,
                            kwargs={
                                "temperature": st.session_state.get("temperature_chat_slider"),
                                "max_tokens": st.session_state.get("max_tokens_chat_slider"),
                            },
                            api_key=st.session_state.get("api_key_chat_input"),
                            base_url=st.session_state.get("base_url_chat_input")
                        ),
                        embedding=LLMConfig(
                            model_type="embedding",
                            institution=st.session_state.get("institution_embed_select"),
                            model=st.session_state.get("model_embed_input"),
                            user_message=user_message,
                            api_key=st.session_state.get("api_key_embed_input"),
                            base_url=st.session_state.get("base_url_embed_input")
                        ),
                        rerank=LLMConfig(
                            model_type="rerank",
                            institution=st.session_state.get("institution_rerank_select"),
                            model=st.session_state.get("model_rerank_input"),
                            api_key=st.session_state.get("api_key_rerank_input"),
                            base_url=st.session_state.get("base_url_rerank_input")
                        ),
                        vl=LLMConfig(
                            model_type="vl",
                            institution=st.session_state.get("institution_vl_select"),
                            model=st.session_state.get("model_vl_input"),
                            kwargs={
                                "temperature": st.session_state.get("temperature_vl_slider"),
                                "max_tokens": st.session_state.get("max_tokens_vl_slider"),
                            },
                            api_key=st.session_state.get("api_key_vl_input"),
                            base_url=st.session_state.get("base_url_vl_input")
                        ),
                        rag=RAGConfig(
                            Query_method=query_method,
                            Question=user_message,
                            top_k=top_k_value,
                            use_Rerank=False,
                            concurrency=1
                        ),
                        database=DatabaseConfig(
                            neo4j=None,  
                            mongo=None, 
                            milvus=None
                        )
                    )

                elif i == 4:  # 检索材料数据
                    if query_method == "RAG智能检索":
                        # 使用 LightRAG 进行检索(去掉多余调用)
                        rag = asyncio.run(initialize_lightrag())
                        query_param = QueryParam(
                            mode="hybrid",
                            top_k=top_k_value,
                            enable_rerank=False  # 明确禁用rerank，避免警告
                        )
                        
                        # 执行 LightRAG 检索
                        async def perform_lightrag_retrieval():
                            try:
                                rag = await initialize_lightrag()

                                # ① 只拿检索上下文（不生成答案）
                                param = QueryParam(mode="hybrid",
                                                top_k=top_k_value,
                                                enable_rerank=False,
                                                only_need_context=True)  # 👈 关键

                                context = await rag.aquery(user_message, param=param)

                                # 智能解析，只保留有实际内容的片段
                                retrieval_results = []
                                if isinstance(context, str) and context.strip():
                                    chunks = [c.strip() for c in context.split("\n\n") if c.strip()]
                                    
                                    # 定义实际内容片段的索引和类型
                                    content_mapping = [
                                        {"index": 1, "type": "entity", "title": "知识图谱实体数据", "base_score": 0.95},
                                        {"index": 3, "type": "relationship", "title": "知识图谱关系数据", "base_score": 0.92},
                                        {"index": 5, "type": "document", "title": "文献文档片段", "base_score": 0.88}
                                    ]
                                    
                                    for mapping in content_mapping:
                                        if mapping["index"] < len(chunks):
                                            chunk = chunks[mapping["index"]]
                                            
                                            # 计算更严谨的相关性分数
                                            # 基于内容长度、关键词匹配、结构完整性等
                                            content_score = calculate_content_relevance(chunk, user_message)
                                            
                                            # 最终分数 = 基础分数 * 内容相关性系数
                                            final_score = mapping["base_score"] * content_score
                                            
                                            retrieval_results.append({
                                                "text": chunk,
                                                "score": round(final_score, 4),
                                                "type": mapping["type"],
                                                "title": mapping["title"],
                                                "metadata": {
                                                    "content_length": len(chunk),
                                                    "has_json": chunk.count('{') > 0,
                                                    "entity_count": chunk.count('"entity"') if mapping["type"] == "entity" else 0,
                                                    "relationship_count": chunk.count('"entity1"') if mapping["type"] == "relationship" else 0,
                                                }
                                            })
                                    
                                    # 按分数排序
                                    retrieval_results.sort(key=lambda x: x['score'], reverse=True)

                                logger.info(f"LightRAG 提取到 {len(retrieval_results)} 个有效内容片段")
                                return retrieval_results

                            except Exception as e:
                                logger.error(f"LightRAG 检索失败: {e}")
                                return []
                        
                        # 执行异步检索
                        retrieval_results = asyncio.run(perform_lightrag_retrieval())
                        
                    else:
                        # 使用原有的 MongoDB 检索
                        retrieval_results = search_mongo_documents(user_message, top_k_value)
                    
                elif i == 5:  # 生成分析提示
                    if query_method == "RAG智能检索":
                        full_prompt = build_RAG_prompt(user_message, retrieval_results)
                    else:
                        full_prompt = build_enhanced_prompt(user_message, retrieval_results)
                    
                elif i == 6:  # 调用AI模型
                    # 显示加载动画
                    show_loading_spinner(progress_placeholder, "AI模型深度分析中，请耐心等待...")
                    # 添加取消按钮
                    cancel_col1, cancel_col2 = st.columns([3, 1])
                    with cancel_col2:
                        if st.button("❌ 取消分析", key="cancel_analysis_btn", use_container_width=True):
                            st.session_state.cancel_requested = True
                            st.warning("分析任务已取消")
                            progress_placeholder.empty()
                            st.stop()
                    
                    # 判断是否使用微调模型（仅在RAG智能检索时强制使用，其他模式根据设置）
                    use_fine_tuned_for_chat = st.session_state.get("use_fine_tuned", False)
                    
                    # 记录使用的模型类型
                    if use_fine_tuned_for_chat:
                        logger.info("使用微调模型进行回答生成")
                    else:
                        logger.info("使用AI模型提供商进行回答生成")
                    
                elif i == 7:  # 获取分析结果
                    # 检查是否请求取消
                    if st.session_state.get("cancel_requested", False):
                        st.warning("分析任务已被用户取消")
                        progress_placeholder.empty()
                        st.stop()
                    
                    # 根据是否使用微调模型选择不同的生成方式
                    if use_fine_tuned_for_chat:
                        # 使用本地微调模型生成回答
                        try:
                            # 构建提示词
                            system_prompt = """你是核电材料领域的专家，拥有深厚的材料科学和核工程知识。
请基于提供的问题和参考数据，提供专业、准确、客观的分析结果。
分析应包括：
1. 基于现有研究和数据的科学结论
2. 不同材料选项的优缺点对比
3. 实际应用中的注意事项和限制条件
4. 相关安全标准和规范的参考
5. 建议的进一步研究方向（如适用）

请使用专业术语，但保持表述清晰易懂，避免过度简化可能导致的误解。
对于不确定的信息，应明确说明并提供可能的误差范围。"""
                            
                            # 获取生成参数
                            temperature = st.session_state.get("temperature_chat_slider", 0.3)
                            max_tokens = st.session_state.get("max_tokens_chat_slider", 4096)
                            
                            # 调用本地微调模型
                            logger.info("正在调用本地微调模型生成回答...")
                            response_text = hf_model_complete(
                                prompt=full_prompt,
                                system_prompt=system_prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=0.9,
                                top_k=50
                            )
                            logger.info("微调模型生成完成")
                        except Exception as e:
                            logger.error(f"微调模型推理失败: {str(e)}")
                            # 如果微调模型失败，回退到AI提供商
                            logger.warning("回退到AI模型提供商...")
                            chat_client = global_config.chat.client
                            response_text = run_async(chat_client.get_response(
                                user_message=full_prompt,
                                model_type="chat"
                            ))
                    else:
                        # 使用AI模型提供商
                        chat_client = global_config.chat.client
                        response_text = run_async(chat_client.get_response(
                            user_message=full_prompt,
                            model_type="chat"
                        ))  
                    
                elif i == 8:  # 整理最终报告
                    # 最终整理逻辑
                    pass
                
                # 模拟处理时间
                time.sleep(step["duration"])
            
            # 完成进度显示
            progress_placeholder.success("✅ 分析完成！")
            time.sleep(1)
            
            # 显示结果
            with result_placeholder.container():
                st.markdown(response_text)
                
                # 如果是 RAG 查询，并且有检索到的文献，才显示额外的信息
                if query_method == "RAG智能检索" and retrieval_results:
                    st.markdown("### 🔍 LightRAG 检索详情")
                    
                    # 创建可视化标签页
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📚 检索结果", 
                        "🕸 知识图谱", 
                        "📊 数据统计", 
                        "🔍 高级分析"
                    ])
                    
                    # ── ① 检索结果卡片 ─────────────────────────────────────────
                    with tab1:
                        if not retrieval_results:
                            # st.info("本次查询未检索到具体片段，结果基于通用知识生成。")
                            pass
                        else:
                                # 自定义CSS样式 - 更现代化的设计
                            st.markdown("""
                            <style>
                            /* 主容器样式 */
                            .retrieval-container {
                                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                            }
                            
                            /* 统计卡片 */
                            .stats-container {
                                display: flex;
                                gap: 16px;
                                margin-bottom: 24px;
                                flex-wrap: wrap;
                            }
                            .stat-card {
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                color: white;
                                padding: 16px 24px;
                                border-radius: 12px;
                                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.25);
                                flex: 1;
                                min-width: 200px;
                            }
                            .stat-label {
                                font-size: 14px;
                                opacity: 0.9;
                                margin-bottom: 4px;
                            }
                            .stat-value {
                                font-size: 28px;
                                font-weight: 700;
                            }
                            
                            /* 标签系统 */
                            .type-badge {
                                display: inline-block;
                                padding: 4px 12px;
                                border-radius: 20px;
                                font-size: 12px;
                                font-weight: 500;
                                margin-right: 8px;
                            }
                            .type-entity { background: #e3f2fd; color: #1565c0; }
                            .type-relationship { background: #f3e5f5; color: #7b1fa2; }
                            .type-chunk { background: #e8f5e8; color: #2e7d32; }
                            .type-reference { background: #fff3e0; color: #ef6c00; }
                            .type-other { background: #f5f5f5; color: #616161; }
                            
                            /* 卡片网格 */
                            .cards-grid {
                                display: grid;
                                grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
                                gap: 20px;
                                margin-top: 20px;
                            }
                            
                            /* 增强卡片样式 */
                            .enhanced-card {
                                background: white;
                                border-radius: 12px;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                                border: 1px solid #eef2f6;
                                overflow: hidden;
                                transition: all 0.2s ease;
                            }
                            .enhanced-card:hover {
                                box-shadow: 0 8px 20px rgba(0,0,0,0.08);
                                transform: translateY(-2px);
                                border-color: #d0d9e8;
                            }
                            
                            /* 卡片头部 */
                            .card-header {
                                padding: 16px;
                                background: #f8fafd;
                                border-bottom: 1px solid #eef2f6;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            }
                            .card-title {
                                font-weight: 600;
                                font-size: 16px;
                                color: #1a2639;
                                margin: 0;
                                display: flex;
                                align-items: center;
                                gap: 8px;
                            }
                            .score-indicator {
                                background: #4caf50;
                                color: white;
                                padding: 4px 8px;
                                border-radius: 20px;
                                font-size: 12px;
                                font-weight: 600;
                            }
                            
                            /* 卡片内容 */
                            .card-body {
                                padding: 16px;
                            }
                            .meta-row {
                                display: flex;
                                gap: 16px;
                                font-size: 12px;
                                color: #64748b;
                                margin-bottom: 12px;
                                flex-wrap: wrap;
                            }
                            .meta-item {
                                display: flex;
                                align-items: center;
                                gap: 4px;
                            }
                            .content-preview {
                                background: #f8fafc;
                                border-radius: 8px;
                                padding: 12px;
                                font-size: 13px;
                                color: #334155;
                                line-height: 1.6;
                                max-height: 150px;
                                overflow: hidden;
                                position: relative;
                                margin-bottom: 12px;
                                border: 1px solid #e9eef2;
                                font-family: 'Monaco', 'Menlo', monospace;
                            }
                            .content-preview::after {
                                content: '';
                                position: absolute;
                                bottom: 0;
                                left: 0;
                                right: 0;
                                height: 40px;
                                background: linear-gradient(transparent, #f8fafc);
                                pointer-events: none;
                            }
                            
                            /* 实体特殊样式 */
                            .entity-grid {
                                display: grid;
                                grid-template-columns: auto 1fr;
                                gap: 8px 12px;
                                background: #f8fafc;
                                padding: 12px;
                                border-radius: 8px;
                                font-size: 13px;
                            }
                            .entity-label {
                                color: #64748b;
                                font-weight: 500;
                            }
                            .entity-value {
                                color: #0f172a;
                            }
                            .entity-highlight {
                                background: #fef9e7;
                                border-left: 3px solid #f1c40f;
                                padding: 8px 12px;
                                margin: 8px 0;
                                border-radius: 0 4px 4px 0;
                            }
                            
                            /* 关系特殊样式 */
                            .relationship-display {
                                background: #f8fafc;
                                border-radius: 8px;
                                padding: 16px;
                                text-align: center;
                            }
                            .relation-nodes {
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                gap: 12px;
                                margin-bottom: 12px;
                            }
                            .relation-node {
                                background: white;
                                padding: 8px 16px;
                                border-radius: 30px;
                                border: 2px solid;
                                font-weight: 500;
                                font-size: 14px;
                            }
                            .node-entity1 { border-color: #3b82f6; color: #1e40af; }
                            .node-entity2 { border-color: #8b5cf6; color: #5b21b6; }
                            .relation-arrow {
                                color: #94a3b8;
                                font-size: 20px;
                            }
                            
                            /* 底部操作区 */
                            .card-footer {
                                padding: 12px 16px;
                                background: #ffffff;
                                border-top: 1px solid #eef2f6;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            }
                            
                            /* 标签页样式 */
                            .stTabs [data-baseweb="tab-list"] {
                                gap: 8px;
                            }
                            .stTabs [data-baseweb="tab"] {
                                border-radius: 20px;
                                padding: 8px 16px;
                                background: #f1f5f9;
                            }
                            .stTabs [aria-selected="true"] {
                                background: #3c6382 !important;
                                color: white !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)

                            # 分析检索结果
                            result_types = {'entity': 0, 'relationship': 0, 'chunk': 0, 'reference': 0, 'other': 0}
                            for res in retrieval_results:
                                title = res.get('title', '').lower()
                                if '实体' in title or 'entity' in title:
                                    result_types['entity'] += 1
                                elif '关系' in title or 'relationship' in title:
                                    result_types['relationship'] += 1
                                elif '文档' in title or 'chunk' in title or 'document' in title:
                                    result_types['chunk'] += 1
                                elif '参考' in title or 'reference' in title:
                                    result_types['reference'] += 1
                                else:
                                    result_types['other'] += 1

                            # 统计卡片
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("总片段数", len(retrieval_results), delta=None)
                            with col2:
                                avg_score = sum(r.get('score', 0) for r in retrieval_results) / len(retrieval_results)
                                st.metric("平均相关度", f"{avg_score:.3f}", delta=None)
                            with col3:
                                high_quality = sum(1 for r in retrieval_results if r.get('score', 0) > 0.8)
                                st.metric("高质量片段", high_quality, delta=f"{high_quality/len(retrieval_results)*100:.0f}%")
                            with col4:
                                st.metric("实体/关系数", f"{result_types['entity']+result_types['relationship']}", delta=None)

                            # 类型分布标签
                            type_cols = st.columns(5)
                            type_labels = ['实体', '关系', '文档块', '参考文献', '其他']
                            type_keys = ['entity', 'relationship', 'chunk', 'reference', 'other']
                            type_colors = ['type-entity', 'type-relationship', 'type-chunk', 'type-reference', 'type-other']
                            
                            for col, label, key, color in zip(type_cols, type_labels, type_keys, type_colors):
                                with col:
                                    if result_types[key] > 0:
                                        st.markdown(f'<span class="type-badge {color}">📊 {label}: {result_types[key]}</span>', 
                                                unsafe_allow_html=True)

                            st.divider()

                            # 按类型分类的标签页
                            tab6, tab7, tab8, tab9, tab10 = st.tabs(["📌 全部片段", "🔷 实体", "🔗 关系", "📄 文档块", "📚 参考文献"])
                            
                            def render_entity_content(text):
                                """渲染实体内容"""
                                try:
                                    # 尝试解析JSON格式的实体
                                    if text.startswith('{') and text.endswith('}'):
                                        entities = []
                                        # 可能包含多个JSON对象
                                        for line in text.strip().split('\n'):
                                            line = line.strip()
                                            if line and (line.startswith('{') or line.startswith('{"entity"')):
                                                try:
                                                    entity = json.loads(line.rstrip(','))
                                                    entities.append(entity)
                                                except:
                                                    pass
                                        
                                        if entities:
                                            for entity in entities:
                                                st.markdown(f"""
                                                <div class="entity-grid">
                                                    <span class="entity-label">🔖 实体：</span>
                                                    <span class="entity-value">{entity.get('entity', '未知')}</span>
                                                    <span class="entity-label">📋 类型：</span>
                                                    <span class="entity-value">{entity.get('type', '未知')}</span>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                desc = entity.get('description', '')
                                                if desc:
                                                    desc = desc.replace('<SEP>', '\n\n')
                                                    st.markdown(f'<div class="entity-highlight">{desc[:300]}{"..." if len(desc)>300 else ""}</div>', 
                                                            unsafe_allow_html=True)
                                                return
                                except:
                                    pass
                                # 默认显示
                                st.text(text[:300] + ('...' if len(text) > 300 else ''))

                            def render_relationship_content(text):
                                """渲染关系内容"""
                                try:
                                    if text.startswith('{') and text.endswith('}'):
                                        relations = []
                                        for line in text.strip().split('\n'):
                                            line = line.strip()
                                            if line and (line.startswith('{') or line.startswith('{"entity1"')):
                                                try:
                                                    rel = json.loads(line.rstrip(','))
                                                    relations.append(rel)
                                                except:
                                                    pass
                                        
                                        if relations:
                                            for rel in relations[:1]:  # 只显示第一个关系作为预览
                                                st.markdown(f"""
                                                <div class="relationship-display">
                                                    <div class="relation-nodes">
                                                        <span class="relation-node node-entity1">{rel.get('entity1', 'A')}</span>
                                                        <span class="relation-arrow">→</span>
                                                        <span class="relation-node node-entity2">{rel.get('entity2', 'B')}</span>
                                                    </div>
                                                    <div style="color: #475569; font-size: 13px; background: white; padding: 8px; border-radius: 6px;">
                                                        {rel.get('description', '无描述')[:150]}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                return
                                except:
                                    pass
                                st.text(text[:300])

                            def render_chunk_content(text):
                                """渲染文档块内容"""
                                try:
                                    if text.startswith('{') and 'content' in text:
                                        data = json.loads(text)
                                        content = data.get('content', '')
                                        if content:
                                            # 提取关键信息
                                            pages = re.findall(r'第\s*(\d+)\s*页', content)
                                            if pages:
                                                st.caption(f"📑 页码: {', '.join(pages)}")
                                            
                                            # 显示预览
                                            content_clean = re.sub(r'!\[.*?\]\(.*?\)', '[图片]', content)
                                            st.markdown(f'<div class="content-preview">{content_clean[:300]}</div>', 
                                                    unsafe_allow_html=True)
                                            return
                                except:
                                    pass
                                st.text(text[:300])

                            def render_reference_content(text):
                                """渲染参考文献内容"""
                                st.info("📚 这是参考文献索引，对应完整文档列表")
                                st.code(text[:500] if len(text) > 500 else text, language='text')

                            # 在对应的标签页中显示
                            tabs = [tab6, tab7, tab8, tab9, tab10]
                            tab_filters = [None, 'entity', 'relationship', 'chunk', 'reference']
                            
                            for tab, filter_type in zip(tabs, tab_filters):
                                with tab:
                                    filtered_results = retrieval_results
                                    if filter_type:
                                        filtered_results = [
                                            r for r in retrieval_results 
                                            if filter_type in r.get('title', '').lower() 
                                            or (filter_type == 'entity' and ('实体' in r.get('title', '') or 'entity' in r.get('title', '').lower()))
                                            or (filter_type == 'relationship' and ('关系' in r.get('title', '') or 'relationship' in r.get('title', '').lower()))
                                            or (filter_type == 'chunk' and ('文档' in r.get('title', '') or 'chunk' in r.get('title', '').lower()))
                                            or (filter_type == 'reference' and ('参考' in r.get('title', '') or 'reference' in r.get('title', '').lower()))
                                        ]
                                    
                                    if not filtered_results:
                                        st.info(f"没有找到{filter_type or '全部'}类型的片段")
                                        continue
                                    
                                    # 网格布局显示卡片
                                    cols = st.columns(2)
                                    for idx, res in enumerate(filtered_results):
                                        with cols[idx % 2]:
                                            title = res.get('title', f'片段 {idx+1}')
                                            score = res.get('score', 0)
                                            text = res.get('text', '')
                                            metadata = res.get('metadata', {})
                                            
                                            # 确定类型标签
                                            type_label = 'other'
                                            type_text = '其他'
                                            if 'entity' in title.lower() or '实体' in title:
                                                type_label = 'entity'
                                                type_text = '实体'
                                            elif 'relationship' in title.lower() or '关系' in title:
                                                type_label = 'relationship'
                                                type_text = '关系'
                                            elif 'chunk' in title.lower() or '文档' in title:
                                                type_label = 'chunk'
                                                type_text = '文档块'
                                            elif 'reference' in title.lower() or '参考' in title:
                                                type_label = 'reference'
                                                type_text = '参考文献'
                                            
                                            # 卡片容器
                                            with st.container():
                                                st.markdown(f"""
                                                <div class="enhanced-card">
                                                    <div class="card-header">
                                                        <div class="card-title">
                                                            <span class="type-badge type-{type_label}">{type_text}</span>
                                                            {title[:40]}
                                                        </div>
                                                        <span class="score-indicator">{score:.3f}</span>
                                                    </div>
                                                    <div class="card-body">
                                                        <div class="meta-row">
                                                            <span class="meta-item">📊 相似度: {score:.4f}</span>
                                                            <span class="meta-item">📁 来源: {metadata.get('source', '未知')}</span>
                                                        </div>
                                                """, unsafe_allow_html=True)
                                                
                                                # 根据类型渲染内容
                                                if type_label == 'entity':
                                                    render_entity_content(text)
                                                elif type_label == 'relationship':
                                                    render_relationship_content(text)
                                                elif type_label == 'chunk':
                                                    render_chunk_content(text)
                                                elif type_label == 'reference':
                                                    render_reference_content(text)
                                                else:
                                                    st.text(text[:200] + ('...' if len(text) > 200 else ''))
                                                
                                                st.markdown(f"""
                                                    </div>
                                                    <div class="card-footer">
                                                        <span style="color: #64748b;">🔍 片段 #{idx+1}</span>
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                # 展开查看完整内容
                                                with st.expander("📖 查看完整内容"):
                                                    st.text(text)
                                                
                                                # 如果是实体或关系，尝试格式化显示
                                                if type_label in ['entity', 'relationship'] and ('```json' in text or '{' in text):
                                                    try:
                                                        # 提取JSON部分
                                                        json_str = text.replace('```json', '').replace('```', '').strip()
                                                        if json_str.startswith('{'):
                                                            st.json(json.loads(json_str))
                                                    except:
                                                        pass

                            # 底部分析
                            with st.expander("📊 检索结果分析"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("分数分布")
                                    scores = [r.get('score', 0) for r in retrieval_results]
                                    st.bar_chart({"分数段": scores})
                                
                                with col2:
                                    st.subheader("类型分布")
                                    type_data = {
                                        "实体": result_types['entity'],
                                        "关系": result_types['relationship'],
                                        "文档块": result_types['chunk'],
                                        "参考文献": result_types['reference'],
                                        "其他": result_types['other']
                                    }
                                    st.bar_chart(type_data)
                     # ── ② 核心知识图谱 ─────────────────────────────────────────
                    # ── ② 核心知识图谱（真实 RAG 实体 & 关系） ────────────────
                    with tab2:
                        st.markdown("#### 🕸 核心实体关系图（RAG 原始结果）")
                        
                        # 知识图谱说明
                        st.markdown("""
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #3c6382;">
                            <h4 style="margin-top: 0; color: #0a3d62;">📊 知识图谱说明</h4>
                            <p style="margin-bottom: 8px;">此图谱基于 RAG 检索结果自动构建，展示核电材料领域的关键实体及其关系：</p>
                            <ul style="margin-bottom: 0;">
                                <li><strong>节点</strong>：代表材料、性能、环境等实体</li>
                                <li><strong>边</strong>：表示实体间的关系，粗细代表关系强度</li>
                                <li><strong>颜色</strong>：区分不同类型的实体</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if not retrieval_results:
                            st.info("❌ 无检索数据，无法生成图谱。")
                        else:
                            # 1. 解析真实实体/关系
                            entities, relations = build_real_kg(retrieval_results)
                            if not entities:
                                st.info("🔍 未提取到实体，无法绘图。")
                            else:
                                # 2. 增强颜色映射和节点大小
                                color_map = {
                                    "material": "#4c72b0",      # 材料 - 蓝色
                                    "property": "#dd8452",      # 性能 - 橙色
                                    "environment": "#55a868",   # 环境 - 绿色
                                    "defect": "#c44e52",        # 缺陷 - 红色
                                    "process": "#8172b3",       # 工艺 - 紫色
                                    "standard": "#ccb974",      # 标准 - 金色
                                    "unknown": "#999999"        # 未知 - 灰色
                                }
                                
                                # 节点大小基于关系数量
                                node_relations = {}
                                for rel in relations:
                                    node_relations[rel[0]] = node_relations.get(rel[0], 0) + 1
                                    node_relations[rel[1]] = node_relations.get(rel[1], 0) + 1
                                
                                nodes = []
                                for e in entities:
                                    rel_count = node_relations.get(e[0], 1)
                                    size = 20 + min(rel_count * 5, 30)  # 基于关系数量调整大小
                                    nodes.append({
                                        "id": e[0], 
                                        "type": e[1], 
                                        "size": size,
                                        "relations": rel_count
                                    })
                                
                                edges = [{"source": r[0], "target": r[1], "label": r[2], "weight": r[3]} for r in relations]

                                # 3. PyVis 画图
                                from pyvis.network import Network
                                net = Network(
                                    height="600px", 
                                    width="100%", 
                                    bgcolor="#ffffff", 
                                    font_color="#333",
                                    notebook=False
                                )
                                
                                # 添加节点
                                for n in nodes:
                                    node_color = color_map.get(n["type"], color_map["unknown"])
                                    node_title = f"""
                                    Type: {n.get("type", "unknown")}
                                    Relations: {n.get("relations", 0)}
                                    ID: {n["id"]}
                                    """
                                    net.add_node(
                                        n["id"], 
                                        label=n["id"],
                                        color=node_color,
                                        size=n["size"],
                                        title=node_title.strip(),
                                        font={"size": 14, "face": "Arial"}
                                    )

                                # 补缺失节点 & 边
                                node_ids = set(net.get_nodes())
                                for e in edges:
                                    for nid in (e["source"], e["target"]):
                                        if nid not in node_ids:
                                            net.add_node(
                                                nid, 
                                                label=nid, 
                                                color=color_map["unknown"], 
                                                size=15, 
                                                title=f"Type: unknown\nID: {nid}"
                                            )
                                            node_ids.add(nid)
                                    # 根据权重设置边的颜色和样式
                                    edge_color = "#888888"
                                    if e["weight"] > 0.8:
                                        edge_color = "#e74c3c"  # 强关系 - 红色
                                    elif e["weight"] > 0.6:
                                        edge_color = "#f39c12"  # 中等关系 - 橙色
                                    
                                    net.add_edge(
                                        e["source"], 
                                        e["target"],
                                        width=e["weight"]*4,
                                        color=edge_color,
                                        title=f"{e['label']} (强度: {e['weight']:.2f})"
                                    )

                                # 设置物理布局和交互选项
                                net.set_options("""
                                var options = {
                                  "physics": {
                                    "enabled": true,
                                    "stabilization": {"iterations": 100},
                                    "barnesHut": {
                                      "gravitationalConstant": -8000,
                                      "centralGravity": 0.3,
                                      "springLength": 95,
                                      "springConstant": 0.04,
                                      "damping": 0.09
                                    }
                                  },
                                  "interaction": {
                                    "hover": true,
                                    "tooltipDelay": 200,
                                    "keyboard": {"enabled": true}
                                  }
                                }
                                """)

                                # 保存并显示图谱
                                net.save_graph("kg.html")
                                with open("kg.html", "r", encoding="utf-8") as f:
                                    html = f.read()
                                st.components.v1.html(html, height=620)

                                # 4. 增强图例和统计信息
                                st.markdown("---")
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown("### 🎨 图例说明")
                                    
                                    # 使用更简单的方式显示图例
                                    st.markdown("#### 实体类型颜色编码")
                                    for tp, col_hex in color_map.items():
                                        st.markdown(
                                            f"<span style='display:inline-block;width:16px;height:16px;background:{col_hex};border-radius:50%;margin-right:8px;'></span> "
                                            f"**{tp.capitalize()}**",
                                            unsafe_allow_html=True
                                        )
                                    
                                    st.markdown("---")
                                    st.markdown("#### 可视化说明")
                                    st.markdown("- **边粗细**: 表示关系强度")
                                    st.markdown("- **节点大小**: 表示连接关系数量")
                                    st.markdown("- **边颜色**: 红色表示强关系，橙色表示中等关系，灰色表示弱关系")
                                
                                with col2:
                                    st.markdown("### 📈 图谱统计")
                                    st.metric("实体数量", len(nodes))
                                    st.metric("关系数量", len(edges))
                                    st.metric("平均关系数", f"{sum(n.get('relations', 0) for n in nodes) / len(nodes):.1f}")
                                
                                # 5. 关键洞察
                                if len(entities) > 0:
                                    st.markdown("### 🔍 关键洞察")
                                    
                                    # 找出中心节点（关系最多的节点）
                                    if node_relations:
                                        central_nodes = sorted(node_relations.items(), key=lambda x: x[1], reverse=True)[:3]
                                        st.markdown("**中心节点（关系最多）:**")
                                        for node, count in central_nodes:
                                            st.markdown(f"- **{node}**: {count} 个关系")
                                    
                                    # 找出强关系
                                    strong_relations = sorted(relations, key=lambda x: x[3], reverse=True)[:3]
                                    if strong_relations:
                                        st.markdown("**强关系（权重最高）:**")
                                        for rel in strong_relations:
                                            st.markdown(f"- **{rel[0]}** → **{rel[1]}**: {rel[3]:.2f} ({rel[2]})")
                    with tab3:
                        st.markdown("#### 📊 RAG 数据统计")
                        
                        try:
                            working_dir = st.session_state.get("RAG_working_dir", os.path.join(project_root, "myKG"))
                            visualizer = LightRAGVisualizer(data_path=working_dir)
                            
                            if visualizer.load_data():
                                # 显示数据概览
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    doc_count = len(visualizer.documents) if visualizer.documents else 0
                                    st.metric("文档数量", doc_count)
                                
                                with col2:
                                    entity_count = len(visualizer.entities) if visualizer.entities else 0
                                    st.metric("实体数量", entity_count)
                                
                                with col3:
                                    relation_count = len(visualizer.relations) if visualizer.relations else 0
                                    st.metric("关系数量", relation_count)
                                
                                with col4:
                                    graph_nodes = len(visualizer.graph_data['nodes']) if visualizer.graph_data else 0
                                    st.metric("图谱节点", graph_nodes)
                                
                                # 生成文档统计
                                if visualizer.documents:
                                    st.markdown("##### 文档统计信息")
                                    visualizer.visualize_document_statistics()
                                
                                # 生成实体网络可视化
                                if visualizer.entities and visualizer.relations:
                                    st.markdown("##### 实体网络")
                                    entity_fig = visualizer.visualize_entity_network()
                                    if entity_fig:
                                        st.plotly_chart(entity_fig, use_container_width=True)
                            else:
                                st.warning("无法加载RAG数据")
                                
                        except Exception as e:
                            st.error(f"数据统计可视化失败: {str(e)}")
                    
                    with tab4:
                        st.markdown("#### 🔍 增强版知识图谱分析")
                        
                        try:
                            working_dir = st.session_state.get("RAG_working_dir", os.path.join(project_root, "myKG"))
                            enhanced_visualizer = EnhancedLightRAGVisualizer(data_path=working_dir)
                            
                            if enhanced_visualizer.load_data():
                                # 生成增强版知识图谱
                                enhanced_fig = enhanced_visualizer.visualize_enhanced_knowledge_graph(max_nodes=25)
                                if enhanced_fig:
                                    st.plotly_chart(enhanced_fig, use_container_width=True)
                                
                                # 显示知识图谱解释
                                explanation = enhanced_visualizer.explain_knowledge_graph()
                                if explanation:
                                    st.markdown("##### 知识图谱解释")
                                    
                                    # 实体分类统计
                                    if explanation.get('entity_categories'):
                                        st.markdown("**实体分类统计:**")
                                        for category, count in explanation['entity_categories'].items():
                                            st.markdown(f"- {category}: {count}个")
                                    
                                    # 关键洞察
                                    if explanation.get('key_insights'):
                                        st.markdown("**关键洞察:**")
                                        for insight in explanation['key_insights']:
                                            st.markdown(f"- {insight}")
                                    
                                    # 生成综合报告
                                    report = enhanced_visualizer.generate_comprehensive_report()
                                    with st.expander("📋 查看完整分析报告"):
                                        st.markdown(report)
                            else:
                                st.warning("无法加载增强版可视化数据")
                                
                        except Exception as e:
                            st.error(f"增强版分析失败: {str(e)}")
                elif query_method == "RAG智能检索" and not retrieval_results:
                    # st.info("🔍 RAG 检索提示：本次查询未检索到具体的相关文献，分析结果基于通用知识生成。")
                    pass
                

        except Exception as e:
            import traceback, sys
            error_msg = "".join(traceback.format_exception(*sys.exc_info()))
            progress_placeholder.error("❌ 分析过程中出现错误")
            log_process(f"处理过程出错: {str(e)}", "系统", process_placeholder, is_error=True)
            with result_placeholder.container():
                st.error(f"错误详情: {error_msg}")

    # 页面底部信息
    st.markdown('<div class="footer">分析结果仅供研究参考，实际应用需遵循核电行业安全标准和规范。</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">© 核电材料智能分析平台 | Version 3.0 </div>', unsafe_allow_html=True)

    return None

# ====================================================#
# 全局模型缓存
@st.cache_resource
def load_model_cached(model_path, use_quantization=True):
    """缓存加载模型，避免重复加载"""
    try:
        logger.info(f"加载模型: {model_path}")
        
        # 量化配置
        quantization_config = None
        if use_quantization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
        )
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"模型加载成功: {model_path}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise e


def hf_model_complete(prompt, history=None, system_prompt=None, **kwargs):
    """HuggingFace 模型完成函数，用于 LightRAG"""
    try:
        # 从 session_state 获取缓存的模型
        model = st.session_state.get("hf_model")
        tokenizer = st.session_state.get("hf_tokenizer")
        
        if model is None or tokenizer is None:
            # 获取合并模型路径
            merged_model_path = st.session_state.get("merged_model_path")
            if not merged_model_path:
                merged_model_path = os.path.join(project_root, "src", "fine_tuning", "output", "nuclear_sft_dapt_swift_fast", "v0-20260310-114209", "merged_model")
            
            logger.info(f"首次加载模型: {merged_model_path}")
            model, tokenizer = load_model_cached(merged_model_path, use_quantization=True)
            st.session_state["hf_model"] = model
            st.session_state["hf_tokenizer"] = tokenizer
        
        # 构建输入
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 生成
        generation_config = {
            "max_new_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        outputs = model.generate(**inputs, **generation_config)
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回复内容
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        elif "assistant" in response:
            response = response.split("assistant")[-1]
            
        return response.strip()
        
    except Exception as e:
        logger.error(f"模型推理失败: {str(e)}")
        raise e


# ====================================================#
# RAG 相关函数
async def initialize_lightrag():
    """初始化 RAG 实例，支持微调模型"""
    try:
        working_dir = st.session_state.get("RAG_working_dir", os.path.join(project_root, "myKG"))
        
        # 获取是否使用微调模型
        use_fine_tuned = st.session_state.get("use_fine_tuned", False)
        
        if use_fine_tuned:
            # 使用合并后的微调模型（直接通过 transformers 加载）
            merged_model_path = os.path.join(project_root, "src", "fine_tuning", "output", "nuclear_sft_dapt_swift_fast", "v0-20260310-114209", "merged_model")
            
            # 将模型路径保存到 session_state
            st.session_state["merged_model_path"] = merged_model_path
            
            logger.info(f"使用合并后的微调模型: {merged_model_path}")
            
            # 使用 HuggingFace 模型函数
            llm_model_func = hf_model_complete
            llm_model_name = "merged_fine_tuned_model"
            
            # 预加载模型到缓存
            try:
                model, tokenizer = load_model_cached(merged_model_path, use_quantization=True)
                st.session_state["hf_model"] = model
                st.session_state["hf_tokenizer"] = tokenizer
                logger.info("微调模型预加载成功")
            except Exception as e:
                logger.warning(f"模型预加载失败，将在首次使用时加载: {e}")
        else:
            # 使用 Ollama 基础模型
            llm_model = st.session_state.get("RAG_llm_model", "qwen2.5:1.5b-instruct-q4_K_S")
            llm_model_func = ollama_model_complete
            llm_model_name = llm_model
            logger.info(f"使用 Ollama 基础模型: {llm_model}")
        
        embed_model = st.session_state.get("RAG_embed_model", "bge-m3:latest")
        ollama_host = st.session_state.get("RAG_ollama_host", "http://127.0.0.1:11434")
        
        # 根据是否使用微调模型选择不同的 LLM 函数
        if use_fine_tuned:
            rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=llm_model_func,
                llm_model_name=llm_model_name,
                llm_model_kwargs={
                    "max_tokens": 2048,
                    "temperature": 0.7,
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=1024,
                    max_token_size=8192,
                    func=lambda texts: ollama_embed(
                        texts=texts, embed_model=embed_model, host=ollama_host
                    ),
                ),
                kv_storage="JsonKVStorage",
                graph_storage="NetworkXStorage",
                vector_storage="NanoVectorDBStorage",
            )
        else:
            rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=ollama_model_complete,
                llm_model_name=llm_model_name,
                llm_model_kwargs={
                    "host": ollama_host,
                    "options": {"num_ctx": 32768},
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=1024,
                    max_token_size=8192,
                    func=lambda texts: ollama_embed(
                        texts=texts, embed_model=embed_model, host=ollama_host
                    ),
                ),
                kv_storage="JsonKVStorage",
                graph_storage="NetworkXStorage",
                vector_storage="NanoVectorDBStorage",
            )

        await rag.initialize_storages()
        await initialize_pipeline_status()
        
        logger.info("RAG 初始化成功")
        return rag
        
    except Exception as e:
        logger.error(f"RAG 初始化失败: {str(e)}")
        raise e

async def get_documents_from_storage(rag, query, top_k):
    """从存储中获取文档"""
    retrieval_results = []
    
    try:
        # 尝试从向量存储获取文档
        try:
            # 获取向量存储
            vector_storage = rag._vector_storage
            if hasattr(vector_storage, 'search'):
                # 搜索相关文档
                search_results = vector_storage.search(query, top_k=top_k)
                for i, result in enumerate(search_results):
                    doc_text = getattr(result, 'text', '') or getattr(result, 'content', '') or str(result)
                    if doc_text and len(doc_text.strip()) > 10:
                        retrieval_results.append({
                            "text": doc_text,
                            "score": getattr(result, 'score', 0.8 - (i * 0.05)),
                            "metadata": getattr(result, 'metadata', {}),
                            "title": f"向量检索文档 {i+1}"
                        })
        except Exception as e:
            logger.warning(f"向量搜索失败: {e}")
        
        # 尝试从知识图谱中获取文档
        try:
            # 获取图存储
            graph_storage = rag._graph_storage
            if hasattr(graph_storage, 'get_graph'):
                graph = graph_storage.get_graph()
                logger.info(f"获取到知识图谱，节点数: {len(graph.nodes)}")
                
                # 获取节点信息
                for i, node_id in enumerate(list(graph.nodes)[:top_k]):
                    node_data = graph.nodes[node_id]
                    node_text = node_data.get('name', '') or node_data.get('text', '') or str(node_data)
                    if node_text and len(node_text.strip()) > 10:
                        retrieval_results.append({
                            "text": node_text,
                            "score": 0.9 - (i * 0.05),
                            "metadata": node_data,
                            "title": f"知识图谱节点 {i+1}"
                        })
        except Exception as e:
            logger.warning(f"获取知识图谱失败: {e}")
        
        # 尝试从键值存储获取文档
        try:
            # 获取键值存储
            kv_storage = rag._kv_storage
            if hasattr(kv_storage, 'get_all'):
                # 获取所有文档
                all_docs = kv_storage.get_all()
                for i, (doc_id, doc_data) in enumerate(list(all_docs.items())[:top_k]):
                    doc_text = doc_data.get('text', '') or doc_data.get('content', '') or str(doc_data)
                    if doc_text and len(doc_text.strip()) > 10:
                        retrieval_results.append({
                            "text": doc_text,
                            "score": 0.7 - (i * 0.05),
                            "metadata": doc_data,
                            "title": f"存储文档 {i+1}"
                        })
        except Exception as e:
            logger.warning(f"获取键值存储失败: {e}")
        
        logger.info(f"从存储中获取到 {len(retrieval_results)} 个文档")
        
    except Exception as e:
        logger.error(f"从存储获取文档失败: {e}")
    
    return retrieval_results

def parse_RAG_response(RAG_response, top_k):
    """解析 RAG 响应，提取检索到的文档信息"""
    retrieval_results = []
    
    # 如果响应是字符串，尝试检查是否包含检索信息
    if isinstance(RAG_response, str):
        # 如果响应是纯字符串，可能只有生成的回答，没有检索文档
        # 但在这种情况下，我们仍然可以尝试从其他来源获取检索信息
        logger.info("RAG 返回字符串响应，可能只有生成内容")
        return []
    
    # 尝试多种方式提取检索到的文档信息
    try:
        # 方式1: 从 retrieved_documents 属性提取
        if hasattr(RAG_response, 'retrieved_documents') and RAG_response.retrieved_documents:
            for i, doc in enumerate(RAG_response.retrieved_documents[:top_k]):
                if hasattr(doc, 'text') and doc.text and len(doc.text.strip()) > 0:
                    retrieval_results.append({
                        "text": doc.text,
                        "score": getattr(doc, 'score', 0.8 - (i * 0.1)),
                        "metadata": getattr(doc, 'metadata', {}),
                        "title": getattr(doc, 'title', f"检索文档 {i+1}")
                    })
        
        # 方式2: 从 chunks 属性提取（如果有）
        if not retrieval_results and hasattr(RAG_response, 'chunks') and RAG_response.chunks:
            for i, chunk in enumerate(RAG_response.chunks[:top_k]):
                if hasattr(chunk, 'text') and chunk.text and len(chunk.text.strip()) > 0:
                    retrieval_results.append({
                        "text": chunk.text,
                        "score": getattr(chunk, 'score', 0.8 - (i * 0.1)),
                        "metadata": getattr(chunk, 'metadata', {}),
                        "title": getattr(chunk, 'title', f"文本块 {i+1}")
                    })
        
        # 方式3: 从 entities 属性提取（如果有）
        if not retrieval_results and hasattr(RAG_response, 'entities') and RAG_response.entities:
            for i, entity in enumerate(RAG_response.entities[:top_k]):
                entity_text = getattr(entity, 'name', '') or getattr(entity, 'text', '')
                if entity_text and len(entity_text.strip()) > 0:
                    retrieval_results.append({
                        "text": entity_text,
                        "score": getattr(entity, 'score', 0.8 - (i * 0.1)),
                        "metadata": getattr(entity, 'metadata', {}),
                        "title": f"实体: {entity_text[:50]}..."
                    })
        
        # 方式4: 尝试从响应对象的其他属性提取
        if not retrieval_results:
            # 检查响应对象是否有其他可能包含文档信息的属性
            for attr_name in ['documents', 'results', 'retrieved_chunks', 'text_chunks']:
                if hasattr(RAG_response, attr_name):
                    docs = getattr(RAG_response, attr_name)
                    if docs and len(docs) > 0:
                        for i, doc in enumerate(docs[:top_k]):
                            text = getattr(doc, 'text', '') or getattr(doc, 'content', '') or str(doc)
                            if text and len(text.strip()) > 0:
                                retrieval_results.append({
                                    "text": text,
                                    "score": getattr(doc, 'score', 0.8 - (i * 0.1)),
                                    "metadata": getattr(doc, 'metadata', {}),
                                    "title": getattr(doc, 'title', f"{attr_name} {i+1}")
                                })
                        break
        
        # 如果仍然没有找到文档，记录详细信息以便调试
        if not retrieval_results:
            logger.warning(f"未能在 RAG 响应中找到文档内容。响应类型: {type(RAG_response)}")
            logger.warning(f"响应对象属性: {dir(RAG_response)}")
            
    except Exception as e:
        logger.error(f"解析 RAG 响应失败: {e}")
        logger.error(f"响应对象: {RAG_response}")
        # 在错误情况下也返回空列表，避免影响主流程
        return []
    
    logger.info(f"成功解析 {len(retrieval_results)} 个检索文档")
    return retrieval_results

def build_RAG_prompt(question, RAG_results):
    """构建 RAG 专用的提示词"""
    retrieved_texts = []
    document_titles = []
    
    for i, result in enumerate(RAG_results):
        text = result.get('text', '') or ''
        metadata = result.get('metadata', {})
        score = result.get('score', 0)
        title = result.get('title', f"文档 {i+1}")
        
        # 提取文档标题
        document_titles.append(f"{i+1}. {title} (相似度: {score:.4f})")
        
        # 确保text不是None，并且有内容
        if text:
            text_preview = text[:800] + '...' if len(text) > 800 else text
        else:
            text_preview = "内容为空"
        
        retrieved_texts.append(f"""
文档 {i+1}: {title} (相似度: {score:.4f})
内容摘要: {text_preview}
        """.strip())
    
    retrieved_content = "\n\n".join(retrieved_texts)
    titles_content = "\n".join(document_titles)
    
    prompt = f"""你是核电材料领域的专家，基于 RAG 检索到的专业文献进行分析。

用户问题: {question}

检索到的相关文献列表:
{titles_content}

详细的文献内容:
{retrieved_content}

请基于以上检索结果，提供专业、准确的分析，要求如下：

1. **文献引用**: 在回答中明确引用相关文献的编号和标题
2. **结合原文**: 基于检索到的具体文献内容进行分析，不要凭空推测
3. **相似度说明**: 对于相似度较高的文献给予更多关注
4. **专业分析**: 提供基于科学数据的材料性能对比和建议

请确保回答专业、客观，并直接引用检索到的文献内容来支持你的分析。
"""
    return prompt

# ====================================================#
# 辅助函数
def log_process(message, module, placeholder, is_error=False):
    """记录处理步骤并更新显示"""
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


def show_advanced_progress(placeholder, current_step, total_steps, message, sub_message=""):
    """显示高级进度指示器"""
    progress = current_step / total_steps
    percent = int(progress * 100)
    
    html_content = f"""
    <div class="progress-container">
        <div class="progress-header">
            <span style="color: #0a3d62;">{message}</span>
            <span style="color: #3c6382;">{percent}%</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {percent}%;"></div>
        </div>
        <div class="progress-subtext">
            {sub_message}
        </div>
    </div>
    """
    placeholder.markdown(html_content, unsafe_allow_html=True)


def show_loading_spinner(placeholder, message):
    """显示旋转加载器"""
    html_content = f"""
    <div style="text-align: center; padding: 30px;">
        <div class="spinner"></div>
        <div style="margin-top: 20px; color: #3c6382; font-weight: bold; font-size: 1.1em;">
            {message}
        </div>
        <div style="margin-top: 10px; color: #666; font-size: 0.9em;">
            这可能需要一些时间，请勿关闭页面...
        </div>
    </div>
    """
    placeholder.markdown(html_content, unsafe_allow_html=True)


def create_cancel_button():
    """创建取消任务按钮"""
    if st.button("❌ 取消任务", key="cancel_btn", use_container_width=True):
        st.session_state.cancel_requested = True
        st.warning("取消请求已发送，正在停止分析任务...")
        # 这里可以添加实际取消逻辑
        st.stop()

def extract_json_lines(text: str):
    """
    把一段 chunk 里所有 ```json ... ``` 或裸 {...} 行抓出来
    返回 list[dict]
    """
    # 1. 代码块 ```json ... ```
    blocks = re.findall(r'```json(.*?)```', text, flags=re.S)
    # 2. 单行 {...}
    lines = re.findall(r'^\s*\{.*?\}\s*$', text, flags=re.M)
    objs = []
    for b in blocks + lines:
        try:
            # 可能有多行 jsonl
            for ln in b.strip().splitlines():
                if ln.strip():
                    objs.append(json.loads(ln))
        except Exception:
            continue
    return objs

def build_real_kg(retrieval_results):
    """
    输入: retrieval_results（perform_RAG_retrieval 返回的 list）
    输出: entities=list[(name, type)], relations=list[(src, dst, label, weight)]
    权重先用相似度，没有就用 0.5
    """
    ent_set, rel_set = set(), set()
    for res in retrieval_results:
        text = res.get("text", "")
        sim  = res.get("score", 0.5)
        for obj in extract_json_lines(text):
            # ----- 实体 -----
            if "entity" in obj and "type" in obj:
                ent_set.add((obj["entity"], obj["type"]))
            # ----- 关系 -----
            if "entity1" in obj and "entity2" in obj:
                src, dst = obj["entity1"], obj["entity2"]
                label = obj.get("description", "relates_to")[:30]  # 太长截断
                rel_set.add((src, dst, label, sim))
    return list(ent_set), list(rel_set)


# ====================================================#
# 结果展示函数（预设计）
def display_material_performance_results(response):
    """展示材料性能分析结果"""
    if not response:
        st.warning("未获取到分析结果")
        return

    st.markdown("### 材料性能综合评估")
    st.markdown("#### 主要发现")
    st.markdown(response.get('summary', '无摘要信息'))

    if 'performance_table' in response:
        st.markdown("#### 材料性能对比")
        st.dataframe(response['performance_table'])

    if 'key_properties' in response:
        st.markdown("#### 关键性能指标")
        for prop, value in response['key_properties'].items():
            st.markdown(f"- **{prop}**: {value}")

    if 'references' in response:
        st.markdown("#### 参考资料")
        for ref in response['references'][:5]:  # 只显示前5条
            st.markdown(f"- {ref}")


def display_radiation_effect_results(response):
    """展示辐射效应评估结果"""
    if not response:
        st.warning("未获取到分析结果")
        return

    st.markdown("### 辐射效应评估报告")

    # 辐射敏感性评分卡片
    if 'radiation_sensitivity' in response:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("短期辐射敏感性", response['radiation_sensitivity'].get('short_term', 'N/A'))
        with col2:
            st.metric("中期辐射敏感性", response['radiation_sensitivity'].get('mid_term', 'N/A'))
        with col3:
            st.metric("长期辐射敏感性", response['radiation_sensitivity'].get('long_term', 'N/A'))

    st.markdown("#### 辐射损伤机制分析")
    st.markdown(response.get('damage_mechanisms', '无相关分析'))

    if 'microstructure_changes' in response:
        st.markdown("#### 微观结构变化预测")
        st.markdown(response['microstructure_changes'])


def display_corrosion_results(response):
    """展示腐蚀行为预测结果"""
    if not response:
        st.warning("未获取到分析结果")
        return

    st.markdown("### 材料腐蚀行为分析")

    if 'corrosion_rate' in response:
        st.markdown("#### 腐蚀速率预测")
        st.line_chart(response['corrosion_rate'])

    st.markdown("#### 腐蚀机理分析")
    st.markdown(response.get('corrosion_mechanisms', '无相关分析'))

    if 'protective_measures' in response:
        st.markdown("#### 防护措施建议")
        for measure in response['protective_measures']:
            st.markdown(f"- {measure}")


def display_lifetime_results(response):
    """展示寿命评估结果"""
    if not response:
        st.warning("未获取到分析结果")
        return

    st.markdown("### 材料寿命评估报告")

    # 寿命预测卡片
    if 'lifetime_estimates' in response:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("预期寿命 (年)", response['lifetime_estimates'].get('expected', 'N/A'))
        with col2:
            st.metric("安全寿命 (年)", response['lifetime_estimates'].get('safe', 'N/A'))

    st.markdown("#### 寿命限制因素分析")
    st.markdown(response.get('limiting_factors', '无相关分析'))

    if 'lifetime_extension' in response:
        st.markdown("#### 寿命延长可能性评估")
        st.markdown(response['lifetime_extension'])


def display_substitution_results(response):
    """展示材料替代方案结果"""
    if not response:
        st.warning("未获取到分析结果")
        return

    st.markdown("### 材料替代方案评估")

    if 'alternative_materials' in response:
        st.markdown("#### 替代材料对比")
        st.dataframe(response['alternative_materials'])

    st.markdown("#### 最佳替代方案推荐")
    st.markdown(response.get('recommendation', '无推荐信息'))

    if 'implementation_challenges' in response:
        st.markdown("#### 实施挑战与解决方案")
        st.markdown(response['implementation_challenges'])


def display_failure_analysis_results(response):
    """展示失效模式分析结果"""
    if not response:
        st.warning("未获取到分析结果")
        return

    st.markdown("### 材料失效模式分析")

    if 'failure_modes' in response:
        st.markdown("#### 潜在失效模式及风险")
        for mode, risk in response['failure_modes'].items():
            st.markdown(f"- **{mode}**: 风险等级 {risk}")

    st.markdown("#### 失效预防建议")
    st.markdown(response.get('prevention_measures', '无相关建议'))

    if 'inspection_strategy' in response:
        st.markdown("#### 检测与监测策略")
        st.markdown(response['inspection_strategy'])

# 辅助函数
def calculate_content_relevance(content: str, query: str) -> float:
    """计算内容相关性分数（0.8-1.2之间）"""
    score = 1.0
    
    # 基于长度的评分
    if len(content) > 1000:
        score *= 1.1
    elif len(content) < 100:
        score *= 0.8
    
    # 基于JSON结构的评分
    if content.count('{') > 5:
        score *= 1.05
    
    # 基于关键词匹配
    query_keywords = set(query.lower().split())
    content_lower = content.lower()
    matched_keywords = sum(1 for kw in query_keywords if kw in content_lower)
    if matched_keywords > 0:
        score *= (1 + matched_keywords * 0.05)
    
    return min(max(score, 0.8), 1.2)

def calculate_entity_relevance(entity: dict, query: str) -> float:
    """计算实体与查询的相关性"""
    score = 0.5
    query_lower = query.lower()
    
    # 实体名称匹配
    if entity.get('entity', '').lower() in query_lower:
        score += 0.3
    
    # 描述匹配
    desc = entity.get('description', '').lower()
    for word in query_lower.split():
        if word in desc:
            score += 0.05
    
    return min(score, 1.0)

def calculate_relationship_relevance(rel: dict, query: str) -> float:
    """计算关系与查询的相关性"""
    score = 0.5
    
    # 实体匹配
    query_lower = query.lower()
    if rel.get('entity1', '').lower() in query_lower:
        score += 0.15
    if rel.get('entity2', '').lower() in query_lower:
        score += 0.15
    
    # 描述匹配
    desc = rel.get('description', '').lower()
    for word in query_lower.split():
        if word in desc:
            score += 0.03
    
    return min(score, 1.0)

def calculate_document_relevance(content: str, query: str) -> float:
    """计算文档与查询的相关性"""
    query_words = set(query.lower().split())
    content_lower = content.lower()
    
    # 计算词频
    matches = sum(content_lower.count(word) for word in query_words)
    score = min(0.5 + matches * 0.05, 1.0)
    
    return score

def highlight_keywords(text: str, query: str) -> str:
    """高亮关键词"""
    for word in set(query.split()):
        if len(word) > 2:  # 只高亮长度大于2的词
            text = text.replace(word, f'<span class="keyword-highlight">{word}</span>')
    return text

def parse_entity_json(text: str) -> list:
    """解析实体JSON"""
    entities = []
    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        for line in clean_text.split('\n'):
            line = line.strip()
            if line and line.startswith('{') and '"entity"' in line:
                try:
                    entity = json.loads(line.rstrip(','))
                    entities.append(entity)
                except:
                    if line.endswith(','):
                        try:
                            entity = json.loads(line[:-1])
                            entities.append(entity)
                        except:
                            pass
    except:
        pass
    return entities

def parse_relationship_json(text: str) -> list:
    """解析关系JSON"""
    relationships = []
    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        for line in clean_text.split('\n'):
            line = line.strip()
            if line and line.startswith('{') and '"entity1"' in line:
                try:
                    rel = json.loads(line.rstrip(','))
                    relationships.append(rel)
                except:
                    if line.endswith(','):
                        try:
                            rel = json.loads(line[:-1])
                            relationships.append(rel)
                        except:
                            pass
    except:
        pass
    return relationships

def parse_chunk_json(text: str) -> list:
    """解析文档块JSON"""
    chunks = []
    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        for line in clean_text.split('\n'):
            line = line.strip()
            if line and line.startswith('{') and '"content"' in line:
                try:
                    chunk = json.loads(line.rstrip(','))
                    chunks.append(chunk)
                except:
                    if line.endswith(','):
                        try:
                            chunk = json.loads(line[:-1])
                            chunks.append(chunk)
                        except:
                            pass
    except:
        pass
    return chunks

# 主程序入口
if __name__ == "__main__":
    setup_web_server()
