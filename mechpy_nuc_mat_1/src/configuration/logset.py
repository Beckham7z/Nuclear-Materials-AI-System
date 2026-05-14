import os
import logging

class LogConfig:
    _initialized = False  # 类变量，防止多次初始化

    def __init__(self, log_dir=None, enable_debug=True):
        
        # if LogConfig._initialized:
        #     # 已初始化则直接返回，不再重复配置
        #     self.logger = logging.getLogger('mechpy')
        #     return
        # 获取当前脚本的绝对路径
        current_script_path = os.path.abspath(__file__)
        # 设置日志目录
        if log_dir is None: 
            script_dir = os.path.dirname(current_script_path)
            log_dir = os.path.join(os.path.dirname(script_dir), 'log')
        os.makedirs(log_dir, exist_ok=True)

        # 创建主 logger
        self.logger = logging.getLogger('mechpy')
        self.logger.setLevel(logging.DEBUG)

        # 清除旧的 handler，防止重复添加
        self.logger.handlers.clear()

        # 创建不同的处理器和格式化器
        handlers = {
            'info': self._create_handler('info', log_dir),
            'warning': self._create_handler('warning', log_dir),
            'error': self._create_handler('error', log_dir),
            'debug': self._create_handler('debug', log_dir), 
        }

        # 添加所有处理器到 logger
        for handler in handlers.values():
            self.logger.addHandler(handler)

        LogConfig._initialized = True  # 标记已初始化

    def _create_handler(self, level, log_dir):
        file_name = f'{level}_m3llm.log'
        file_path = os.path.join(log_dir, file_name)
        handler = logging.FileHandler(file_path, mode='w', encoding='utf-8')
        if level == 'error':
            handler.setLevel(logging.ERROR)
            handler.addFilter(lambda record: record.levelno >= logging.ERROR)
            formatter = logging.Formatter(
                '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        elif level == 'warning':
            handler.setLevel(logging.WARNING)
            handler.addFilter(lambda record: record.levelno == logging.WARNING)
            formatter = logging.Formatter(
                '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        elif level == 'debug':
            handler.setLevel(logging.DEBUG)
            handler.addFilter(lambda record: record.levelno == logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s.%(msecs)03d - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:  # info
            handler.setLevel(logging.INFO)
            handler.addFilter(lambda record: record.levelno == logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler.setFormatter(formatter)
        return handler

# 创建全局logger实例
logger = LogConfig(enable_debug=True).logger

# 使用示例
if __name__ == "__main__":
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    logger.debug("这是一条调试日志")