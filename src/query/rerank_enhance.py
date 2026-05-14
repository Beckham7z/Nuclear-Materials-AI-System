from configuration.logset import logger
from configuration.global_config import GlobalConfig
from llm.async_utils import run_async


async def rerank_enhance(global_config: GlobalConfig, rerank_needed_list: list[str]) -> list:
    """异步函数：输入字符串列表，输出重排序后的分数列表"""
    if not global_config.rag.use_Rerank:
        logger.warning("Rerank is not enabled, returning original list.")
        return sorted(range(len(rerank_needed_list)), reverse=True)
    
    # 构造 [[question, answer1], [question, answer2], ...]
    rerank_input = [[global_config.rag.Question, item] for item in rerank_needed_list]
    logger.debug(f"Rerank needed list: {rerank_input}")

    # 调用 LLM rerank（实际会走 _get_rerank_response）
    scores = await global_config.rerank.client.get_response("rerank", user_message=rerank_input)
    return scores