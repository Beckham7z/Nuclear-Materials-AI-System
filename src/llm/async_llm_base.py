import asyncio
from configuration.logset import logger

class AsyncLLMBaseClient:
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.client = None
        self._get_client()

    async def get_response(self, model_type, **kwargs):
        method_map = {
            "chat": self._get_chat_response,
            "embedding": self._get_embedding_response,
            "vl": self._get_vl_response,
            "rerank": self._get_rerank_response,
            "generate": getattr(self, "_get_generate_response", None)
        }
        method = method_map.get(model_type)
        if method is None or not callable(method):
            raise ValueError(f"{self.__class__.__name__} unsupported model type : {model_type}")
        return await method(**kwargs)

    def update_llmconfig(self, key, value):
        allowed_keys = {"user_message", "prompt", "kwargs"}
        if key in allowed_keys:
            setattr(self.llm_config, key, value)
        else:
            logger.warning(f"只允许更新 user_message、prompt、kwargs，不能更新 '{key}'")

    def _get_client(self):
        raise NotImplementedError("子类需实现 _get_client 方法")

    async def _get_chat_response(self, **kwargs):
        raise NotImplementedError("子类需实现 _get_chat_response 方法")

    async def _get_embedding_response(self, **kwargs):
        raise NotImplementedError("子类需实现 _get_embedding_response 方法")

    async def _get_vl_response(self, **kwargs):
        raise NotImplementedError("子类需实现 _get_vl_response 方法")

    async def _get_rerank_response(self, **kwargs):
        raise NotImplementedError("子类需实现 _get_rerank_response 方法")
