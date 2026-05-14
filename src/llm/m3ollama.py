# -*- coding: utf-8 -*-
import asyncio
import ollama
from typing import List, Any
from configuration.logset import logger
from .async_llm_base import AsyncLLMBaseClient


class M3OllamaClient(AsyncLLMBaseClient):
    def _get_client(self):
        base_url = getattr(self.llm_config, "base_url", None)
        self.client = ollama.Client(host=base_url) if base_url else ollama.Client()
        logger.info(f"Ollama {self.llm_config.model_type} client created")

    async def _get_chat_response(self, user_message):
        model = getattr(self.llm_config, "model", "")
        def chat():
            response = self.client.chat(
                model=model,
                messages=[{"role": "user", "content": user_message}]
            )
            return response['message']['content']
        return await asyncio.to_thread(chat)

    async def _get_embedding_response(self, input_list):
        model = getattr(self.llm_config, "model", "")
        if isinstance(input_list, str):
            input_list = [input_list]
        def embed():
            response = self.client.embed(model=model, input=input_list)
            logger.info(f"使用模型: {model}, user_message: {input_list}")
            return response["embeddings"]
        return await asyncio.to_thread(embed)

    async def _get_vl_response(self, user_message, image_data=None):
        model = getattr(self.llm_config, "model", "")
        def chat_vl():
            response = self.client.chat(
                model=model,
                messages=[{"role": "user", "content": user_message}],
                images=[image_data] if image_data else None
            )
            return response['message']['content']
        return await asyncio.to_thread(chat_vl)

    async def _get_rerank_response(self, *args, **kwargs):
        logger.error("Ollama rerank model is not implemented.")
        raise NotImplementedError("Ollama rerank model is not implemented.")