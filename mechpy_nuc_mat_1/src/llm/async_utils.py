import asyncio
import nest_asyncio
import functools
import inspect
import threading

def ensure_async(func):
    """
    装饰器：确保同步函数也能异步调用，自动用asyncio.to_thread包裹。
    """
    if inspect.iscoroutinefunction(func):
        return func
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper


def run_async(coro):
    """
    全局入口：自动检测事件循环环境，安全运行异步任务。
    用法：run_async(some_async_func(...))
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        if threading.current_thread() is threading.main_thread():
            # 主线程已有事件循环，使用 nest_asyncio 兼容后阻塞执行
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            # 子线程，使用 run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
    else:
        return asyncio.run(coro)