# -*- coding: utf-8 -*-

import asyncio
from typing import List,Optional, Dict
from configuration.logset import logger
from .async_llm_base import AsyncLLMBaseClient


class M3FlagEmbedClient(AsyncLLMBaseClient):
    def _get_client(self):
        model_name = getattr(self.llm_config, "model", "")
        FlagReranker = __import__("FlagEmbedding").FlagReranker
        self.client = FlagReranker(model_name, use_fp16=True)
        logger.info(f"FlagEmbed {self.llm_config.model_type} client created")

    async def _get_chat_response(self, *args, **kwargs):
        logger.error("FlagEmbed chat model is not implemented.")
        raise NotImplementedError("FlagEmbed chat model is not implemented.")
    
    async def _get_embedding_response(self, *args, **kwargs):
        logger.error("FlagEmbed embedding model is not implemented.")
        raise NotImplementedError("FlagEmbed embedding model is not implemented.")

    async def _get_rerank_response(self, user_message):
        if not isinstance(user_message, list) or not all(isinstance(item, list) and len(item) == 2 for item in user_message):
            logger.warning(f"Rerank LLM requires user_message to be a nested list of [question, answer]. {user_message}")
            return []
        scores = await asyncio.to_thread(self.client.compute_score, user_message)
        return scores
    
    async def _get_vl_response(self, *args, **kwargs):
        logger.error("FlagEmbed vl model is not implemented.")
        raise NotImplementedError("FlagEmbed vl model is not implemented.")