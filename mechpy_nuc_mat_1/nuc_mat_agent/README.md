# Nuclear Material Agent

基于 LangGraph 的智能核电材料分析系统

## 功能特性

- 🤖 **多步推理**: 使用 LangGraph 实现复杂的任务分解和执行流程
- 🔄 **人机协同**: 支持在关键步骤进行人工审核和反馈
- 🧠 **自我反思**: Agent 自动评估答案质量并进行改进
- 📚 **知识更新**: 支持将高质量问答回流入知识库
- 🔍 **RAG 检索**: 集成 LightRAG 进行混合检索增强生成
- 🕸️ **知识图谱**: 支持知识图谱查询和实体关系分析

## 系统架构

```
nuc_mat_agent/
├── __init__.py          # 包初始化
├── agent.py             # LangGraph Agent 核心逻辑
├── config.yaml          # 配置文件
├── llm_client.py        # LLM 客户端封装
├── prompts.py           # Agent 节点定义
├── rag_client.py        # RAG 检索模块
├── state.py             # Agent 状态定义
├── web_ui.py            # Streamlit 前端界面
├── main.py              # 主程序入口
└── requirements.txt     # 依赖列表
```

## 前端界面布局

- **左侧栏**:
  - 模型配置 (MiniMax/DeepSeek/Ollama/Zhipu)
  - Agent 功能开关 (人机协同、自我反思、知识更新)
  - RAG 配置
  - 聊天记录

- **主体区域**:
  - Agent 执行过程展示
    - 任务分解与步骤
    - Agent 思考过程
    - RAG 检索结果
    - 自我反思结果
  - 聊天界面

## 安装和运行

### 1. 安装依赖

```bash
pip install -r nuc_mat_agent/requirements.txt
```

### 2. 配置

编辑 `nuc_mat_agent/config.yaml` 文件，配置您的 API Key:

```yaml
llm:
  minimax:
    api_key: "your-api-key-here"
```

### 3. 运行

```bash
streamlit run nuc_mat_agent/main.py
```

## 任务类型

系统支持以下类型的任务:

- `standard_matching`: 标准匹配
- `material_comparison`: 材料对比
- `performance_analysis`: 性能分析
- `literature_review`: 文献综述
- `material_substitution`: 材料替代
- `learning_guidance`: 学习指导
- `general_qa`: 通用问答

## Agent 执行流程

```
用户输入
    ↓
任务理解与分解
    ↓
RAG 检索
    ↓
知识图谱查询
    ↓
综合分析
    ↓
答案生成
    ↓
(可选) 人机协同
    ↓
自我反思
    ↓
知识更新
    ↓
返回结果
```

## 示例问题

- "ODS钢的高温性能评价有哪些相关标准？"
- "对比一下 316L 不锈钢和 ODS 钢的抗辐照性能"
- "核反应堆压力容器材料需要满足哪些性能要求？"

## 注意事项

1. 确保 Ollama 服务已启动（用于 Embedding）
2. 确保 RAG 工作目录存在且包含知识库数据
3. MiniMax API Key 需要从 MiniMax 平台申请
