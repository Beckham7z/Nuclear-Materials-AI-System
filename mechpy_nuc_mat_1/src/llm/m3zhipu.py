import asyncio
import time
from zhipuai import ZhipuAI
from configuration.logset import logger
from .async_llm_base import AsyncLLMBaseClient




class M3ZhipuClient(AsyncLLMBaseClient):
    def _get_client(self):
        api_key = getattr(self.llm_config, "api_key", "")
        self.client = ZhipuAI(api_key=api_key)
        logger.info(f"ZhipuAI {self.llm_config.model_type} client created")

    async def _get_chat_response(self, user_message, prompt=None, kwargs=None):
        if kwargs is None:
            kwargs = {}
        model = getattr(self.llm_config, "model", "")
        prompt = prompt if prompt is not None else getattr(self.llm_config, "prompt", "")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ]
        def chat():
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            message = response.choices[0].message
            logger.debug(f"使用模型: {model}, prompt: {prompt}, user_message: {user_message}, kwargs: {kwargs}, response: {message}")
            if hasattr(message, "tool_calls") and message.tool_calls:
                func_name = message.tool_calls[0].function.name
                if func_name == "get_current_time":
                    result = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    logger.info(f"本地函数调用结果: {result}")
                    return result
                else:
                    logger.info(f"未实现的函数: {func_name}")
                    return ""
            else:
                return message.content.strip() if hasattr(message, "content") else ""
        return await asyncio.to_thread(chat)

    async def _get_embedding_response(self, input_data):
        model = getattr(self.llm_config, "model", "")
        if isinstance(input_data, str):
            input_data = [input_data]
        def embed():
            response = self.client.embeddings.create(
                model=model,
                input=input_data
            )
            return [item["embedding"] for item in response.get("data", [])]
        return await asyncio.to_thread(embed)

    async def _get_vl_response(self, *args, **kwargs):
        logger.error("ZhiPu vl model is not implemented.")
        raise ValueError("ZhiPu get_vl_response is not implemented yet.")

    async def _get_rerank_response(self, *args, **kwargs):
        logger.error("ZhiPu rerank model is not implemented.")
        raise ValueError("ZhiPu get_rerank_response is not implemented yet.")


