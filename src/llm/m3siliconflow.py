# -*- coding: utf-8 -*-
import httpx
import asyncio
from typing import List, Any
from configuration.logset import logger
from .async_llm_base import AsyncLLMBaseClient


class M3SiliconFlowClient(AsyncLLMBaseClient):
    def _get_client(self):
        api_key = getattr(self.llm_config, "api_key", "")
        base_url = getattr(self.llm_config, "base_url", "https://api.siliconflow.cn/v1")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self._base_url = base_url
        self._headers = headers
        logger.info(f"SiliconFlow {self.llm_config.model_type} client created")

    async def _get_chat_response(self, user_message):
        model = getattr(self.llm_config, "model", "")
        url = f"{self._base_url}/chat/completions"
        payload = {"model": model, "messages": [{"role": "user", "content": user_message}]}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=self._headers)
            response.raise_for_status()
            data = response.json()
        try:
            return data['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"SiliconFlow chat API 返回格式异常: {data}, error: {e}")
            return ""

    async def _get_embedding_response(self, input_data):
        model = getattr(self.llm_config, "model", "")
        url = f"{self._base_url}/embeddings"
        payload = {"model": model, "input": input_data}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=self._headers)
            response.raise_for_status()
            data = response.json()
        embeddings = [item.get('embedding', []) for item in data.get('data', [])]
        if not all(isinstance(e, list) for e in embeddings):
            logger.warning(f"SiliconFlow embedding API 返回格式异常: {data}")
            return []
        return embeddings

    async def _get_vl_response(self, *args, **kwargs):
        logger.error("SiliconFlow vl model is not implemented.")
        raise NotImplementedError("SiliconFlow vl model is not implemented.")

    async def _get_rerank_response(self, user_message):
        model = getattr(self.llm_config, "model", "")
        if not isinstance(user_message, list) or not all(isinstance(item, list) and len(item) == 2 for item in user_message):
            logger.warning(f"Rerank LLM requires user_message to be a nested list of [question, answer]. {user_message}")
            return []
        question = user_message[0][0]
        documents = [item[1] for item in user_message]
        url = f"{self._base_url}/rerank"
        payload = {"model": model, "query": question, "documents": documents}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=self._headers)
                response.raise_for_status()
                data = response.json()
            scores = [item.get('score', 0) for item in data.get('results', [])]
            return scores
        except Exception as e:
            logger.error(f"SiliconFlow rerank API 调用失败: {e}")
            return []