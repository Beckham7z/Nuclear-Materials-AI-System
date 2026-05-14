# llm/registry.py
# 统一注册所有 LLM 推理客户端类
from llm.m3zhipu import M3ZhipuClient
from llm.m3openai import M3OpenAIClient
from llm.m3ollama import M3OllamaClient
from llm.m3flagembed import M3FlagEmbedClient
from llm.m3xinference import M3XinferenceClient
from llm.m3siliconflow import M3SiliconFlowClient


INSTITUTION_CLIENT_MAP = {
    "zhipu": M3ZhipuClient,
    "openai": M3OpenAIClient,
    "deepseek": M3OpenAIClient,
    "ollama": M3OllamaClient,
    "flagembed": M3FlagEmbedClient,
    "xinference": M3XinferenceClient,
    "siliconflow": M3SiliconFlowClient,
}
