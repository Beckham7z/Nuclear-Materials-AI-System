import asyncio
from typing import List, Any
from xinference.client import RESTfulClient,Client
from configuration.logset import logger
from .async_llm_base import AsyncLLMBaseClient

class M3XinferenceClient(AsyncLLMBaseClient):
    def _get_client(self):
        if self.client:
            return self.client
        base_url = getattr(self.llm_config, "base_url", "")
        api_key = getattr(self.llm_config, "api_key", "")
        if self.llm_config.model_type == "chat":
            self.client = RESTfulClient(base_url=base_url)
        if self.llm_config.model_type == "embedding" or self.llm_config.model_type == "rerank":
            self.client = Client(base_url=base_url, api_key=api_key)
        else:
            raise ValueError(f"Xinference unsupported model type : {self.llm_config.model_type}")
        logger.info(f"Xinference {self.llm_config.model_type} client created")

    async def _get_chat_response(self, user_message, system_message=None, kwargs=None):
        if kwargs is None:
            kwargs = {}
        model = getattr(self.llm_config, "model", "")
        system_message = system_message if system_message is not None else getattr(self.llm_config, "system_message", "")
        def chat():
            return self.client.get_model(model).chat(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                generate_config=kwargs
            ).choices[0].message.content
        return await asyncio.to_thread(chat)

    async def _get_generate_response(self, user_message, kwargs=None):
        if kwargs is None:
            kwargs = {}
        model = getattr(self.llm_config, "model", "")
        if not isinstance(user_message, str):
            raise ValueError("Xinference generate model only supports string input for user_message.")
        def generate():
            return self.client.get_model(model).generate(
                prompt=user_message,
                generate_config=kwargs
            ).choices[0].text
        return await asyncio.to_thread(generate)

    async def _get_embedding_response(self, input_list):
        model = getattr(self.llm_config, "model", "")
        if isinstance(input_list, list):
            input_data = input_list
        else:
            input_data = [input_list]
        def get_embeds():
            embeddings = []
            for text in input_data:
                resp = self.client.get_model(model).create_embedding(text)
                if isinstance(resp, dict) and "data" in resp and len(resp["data"]) > 0:
                    embeddings.append(resp["data"][0]["embedding"])
                else:
                    embeddings.append(None)
            return embeddings
        return await asyncio.to_thread(get_embeds)

    async def _get_vl_response(self, *args, **kwargs):
        raise ValueError("Xinference get_vl_response is not implemented yet.")

    async def _get_rerank_response(self, user_message):
        model = getattr(self.llm_config, "model", "")
        if not isinstance(user_message, list) or not all(isinstance(item, list) and len(item) == 2 for item in user_message):
            logger.warning(f"Xinference rerank model requires user_message to be a nested list of [query, corpus]. user_message: {user_message}")
            return []
        if len(user_message) == 0:
            return []
        query = user_message[0][0]
        candidates = [item[1] for item in user_message]
        def rerank():
            resp = self.client.get_model(model).rerank(candidates, query)
            if isinstance(resp, dict) and "results" in resp:
                scores = [item.get("relevance_score", 0) for item in resp["results"]]
                return scores
            else:
                logger.warning(f"Xinference rerank API 返回格式异常: {resp}")
                return []
        return await asyncio.to_thread(rerank)

    def _xinference_info(self):
        logger.info(f"Xinference model running: {self.client.list_models()} model at {self.llm_config.base_url}")
        return f"Xinference Client: {self.llm_config.model_type} model at {self.llm_config.base_url}"

        

# Example usage:
# pass