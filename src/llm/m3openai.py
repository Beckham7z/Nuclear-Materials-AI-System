# -*- coding: utf-8 -*-
import asyncio
from openai import OpenAI
from typing import List,Optional, Dict
from configuration.logset import logger
from .async_llm_base import AsyncLLMBaseClient


class M3OpenAIClient(AsyncLLMBaseClient):
    def _get_client(self):
        api_key = getattr(self.llm_config, "api_key", "")
        base_url = getattr(self.llm_config, "base_url", None)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"OpenAI {self.llm_config.model_type} client created")

    async def _get_chat_response(self, user_message, kwargs=None):
        if kwargs is None:
            kwargs = {}
        model = getattr(self.llm_config, "model", "")
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": user_message}],
            **kwargs
        )
        message = response.choices[0].message.content
        logger.debug(f"使用模型: {model}, user_message: {user_message}, kwargs: {kwargs}, response: {message}")
        return message

    async def _get_embedding_response(self, input_list):
        model = getattr(self.llm_config, "model", "")
        if isinstance(input_list, str):
            input_list = [input_list]
        response = await asyncio.to_thread(
            self.client.embeddings.create,
            model=model,
            input=input_list
        )
        return [item["embedding"] for item in response.get("data", [])]

    async def _get_vl_response(self, *args, **kwargs):
        logger.error("OpenAI vl model is not implemented.")
        raise NotImplementedError("OpenAI vl model is not implemented.")

    async def _get_rerank_response(self, *args, **kwargs):
        logger.error("OpenAI rerank model is not implemented.")
        raise NotImplementedError("OpenAI rerank model is not implemented.")

