"""
LLM 客户端封装 - 支持 MiniMax OpenAI 兼容接口
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from typing import Dict, Any, Optional, List, Iterator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from datetime import datetime
import json
import yaml

from src.configuration.logset import logger


class LLMClient:
    """LLM 客户端封装 - 支持 MiniMax"""
    
    def __init__(self, config_path: str = None):
        """初始化 LLM 客户端"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        self.config = self._load_config(config_path)
        self.llm_config = self.config.get('llm', {})
        self.agent_config = self.config.get('agent', {})
        
        # 当前选择的模型
        self.current_provider = "minimax"
        self.model = None
        self._initialize_llm()
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}, 使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'llm': {
                'minimax': {
                    'model': 'MiniMax-M2.5',
                    'api_key': '',
                    'base_url': 'https://api.minimaxi.com/v1'
                }
            },
            'agent': {
                'temperature': 1.0,
                'max_tokens': 4096,
                'max_iterations': 10
            }
        }
    
    def _initialize_llm(self):
        """初始化 LLM 模型"""
        try:
            provider = self.current_provider
            if provider in self.llm_config:
                config = self.llm_config[provider]
                
                if provider == "minimax":
                    # MiniMax OpenAI 兼容接口
                    self.model = ChatOpenAI(
                        model=config.get('model', 'MiniMax-M2.5'),
                        api_key=config.get('api_key', ''),
                        base_url=config.get('base_url', 'https://api.minimaxi.com/v1'),
                        temperature=self.agent_config.get('temperature', 1.0),
                        max_tokens=self.agent_config.get('max_tokens', 4096)
                    )
                    self.enable_reasoning_split = config.get('enable_reasoning_split', True)
                elif provider == "deepseek":
                    self.model = ChatOpenAI(
                        model=config.get('model', 'deepseek-chat'),
                        api_key=config.get('api_key', ''),
                        base_url=config.get('base_url', 'https://api.deepseek.com'),
                        temperature=self.agent_config.get('temperature', 0.3),
                        max_tokens=self.agent_config.get('max_tokens', 4096)
                    )
                elif provider == "ollama":
                    from langchain_ollama import ChatOllama
                    self.model = ChatOllama(
                        model=config.get('model', 'qwen2.5:1.5b'),
                        base_url=config.get('base_url', 'http://localhost:11434'),
                        temperature=self.agent_config.get('temperature', 0.3)
                    )
                elif provider == "zhipu":
                    self.model = ChatOpenAI(
                        model=config.get('model', 'glm-4'),
                        api_key=config.get('api_key', ''),
                        base_url=config.get('base_url', 'https://open.bigmodel.cn'),
                        temperature=self.agent_config.get('temperature', 0.3),
                        max_tokens=self.agent_config.get('max_tokens', 4096)
                    )
                logger.info(f"LLM 初始化成功: {provider}")
            else:
                logger.warning(f"未找到 provider 配置: {provider}")
        except Exception as e:
            logger.error(f"LLM 初始化失败: {e}")
    
    def set_provider(self, provider: str):
        """设置 LLM 提供商"""
        self.current_provider = provider
        self._initialize_llm()
    
    async def ainvoke(self, messages: List[BaseMessage]) -> Dict[str, str]:
        """异步调用 LLM，返回思考过程和回答"""
        try:
            if self.model is None:
                return {"reasoning": "", "content": "LLM 模型未初始化"}
            
            # 对于 MiniMax，可以使用 reasoning_split
            if self.current_provider == "minimax" and self.enable_reasoning_split:
                # 直接调用，使用模型的原生格式
                response = await self.model.ainvoke(messages)
                return {
                    "reasoning": "",
                    "content": response.content
                }
            else:
                response = await self.model.ainvoke(messages)
                return {
                    "reasoning": "",
                    "content": response.content
                }
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return {"reasoning": "", "content": f"LLM 调用失败: {str(e)}"}
    
    def invoke(self, messages: List[BaseMessage]) -> Dict[str, str]:
        """同步调用 LLM"""
        try:
            if self.model is None:
                return {"reasoning": "", "content": "LLM 模型未初始化"}
            
            response = self.model.invoke(messages)
            return {
                "reasoning": "",
                "content": response.content
            }
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return {"reasoning": "", "content": f"LLM 调用失败: {str(e)}"}
    
    def stream(self, messages: List[BaseMessage]) -> Iterator[Dict[str, Any]]:
        """流式调用 LLM"""
        try:
            if self.model is None:
                yield {"type": "error", "content": "LLM 模型未初始化"}
                return
            
            for chunk in self.model.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield {"type": "content", "content": chunk.content}
                if hasattr(chunk, 'reasoning_details') and chunk.reasoning_details:
                    for detail in chunk.reasoning_details:
                        if 'text' in detail:
                            yield {"type": "reasoning", "content": detail['text']}
        except Exception as e:
            logger.error(f"LLM 流式调用失败: {e}")
            yield {"type": "error", "content": f"LLM 调用失败: {str(e)}"}
    
    def chat(self, user_message: str, system_prompt: str = None) -> str:
        """简单的对话接口"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_message))
        result = self.invoke(messages)
        return result.get("content", "")
    
    def chat_with_reasoning(self, user_message: str, system_prompt: str = None) -> Dict[str, str]:
        """带思考过程的对话"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_message))
        return self.invoke(messages)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        config = self.llm_config.get(self.current_provider, {})
        return {
            "provider": self.current_provider,
            "model": config.get('model', 'unknown'),
            "temperature": self.agent_config.get('temperature', 1.0),
            "max_tokens": self.agent_config.get('max_tokens', 4096),
            "reasoning_split": config.get('enable_reasoning_split', True) if self.current_provider == "minimax" else False
        }


# ============================================
# RAG 提示词模板
# ============================================
NUCLEAR_MATERIAL_SYSTEM_PROMPT = """你是核电材料领域的专家，拥有深厚的材料科学和核工程知识。

你的职责包括：
1. 基于提供的问题和参考数据，提供专业、准确、客观的分析结果
2. 分析应包括：
   - 基于现有研究和数据的科学结论
   - 不同材料选项的优缺点对比
   - 实际应用中的注意事项和限制条件
   - 相关安全标准和规范的参考
   - 建议的进一步研究方向（如适用）
3. 请使用专业术语，但保持表述清晰易懂
4. 对于不确定的信息，应明确说明并提供可能的误差范围

当前支持的任务类型：
- standard_matching: 标准匹配
- material_comparison: 材料对比
- performance_analysis: 性能分析
- literature_review: 文献综述
- material_substitution: 材料替代
- learning_guidance: 学习指导
- general_qa: 通用问答
"""


# ============================================
# Agent 提示词
# ============================================
AGENT_TASK_DECOMPOSE_PROMPT = """你是一个核电材料领域的任务规划专家。

用户问题：{user_input}

请分析这个问题，并确定：
1. 任务类型（standard_matching, material_comparison, performance_analysis, literature_review, material_substitution, learning_guidance, general_qa）
2. 需要执行的子任务步骤
3. 每个步骤需要调用的工具

请以 JSON 格式返回：
{{
    "task_type": "任务类型",
    "task_steps": [
        {{"step": 1, "name": "步骤名称", "tool": "工具名称", "description": "步骤描述"}}
    ]
}}
"""


AGENT_REFLECTION_PROMPT = """你是一个核电材料领域的质量审核专家。

请评估以下回答的质量：

回答内容：{answer}

原始问题：{question}

请评估：
1. 回答是否准确回答了用户问题
2. 回答是否引用了相关的文献和数据
3. 回答是否存在任何错误或遗漏
4. 回答的专业性和完整性如何

请以 JSON 格式返回：
{{
    "quality_score": 0.0-1.0,
    "needs_improvement": true/false,
    "issues": ["问题列表"],
    "suggestions": ["改进建议"]
}}
"""


AGENT_HUMAN_FEEDBACK_PROMPT = """用户提供了以下反馈：

反馈内容：{feedback}

原始回答：{original_answer}

请根据用户反馈，说明系统将如何调整回答。
"""


# 全局 LLM 客户端实例
_global_llm_client = None


def get_llm_client(config_path: str = None) -> LLMClient:
    """获取全局 LLM 客户端"""
    global _global_llm_client
    if _global_llm_client is None:
        _global_llm_client = LLMClient(config_path)
    return _global_llm_client
