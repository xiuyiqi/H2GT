import logging
from pathlib import Path
from typing import Tuple
from datetime import datetime

# 定义了日志的输出格式
def get_formatter() -> logging.Formatter:
    return logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')

# 初始化logger
def initialize_logger() -> logging.Logger:
    logger = logging.getLogger() # 获取默认的logger
    logger.setLevel(logging.INFO) # 设置级别
    for handler in logger.handlers: # 关闭并清空现有的处理器
        handler.close()
    logger.handlers.clear()

    formatter = get_formatter() # 日志格式
    stream_handler = logging.StreamHandler() # 流处理器
    stream_handler.setFormatter(formatter) # 设置格式
    logger.addHandler(stream_handler) # 将流处理器添加到logger

    return logger

# 设置一个文件日志处理器
def set_file_handler(log_file_path: Path) -> logging.Logger:
    logger = initialize_logger() # 初始化logger
    formatter = get_formatter() # 日志格式
    file_handler = logging.FileHandler(str(log_file_path)) # 文件处理器
    file_handler.setFormatter(formatter) # 设置格式
    logger.addHandler(file_handler) # 将文件处理器添加到logger

    return logger

# 在指定路径加载日志
def logger_factory() -> Tuple[logging.Logger]:
    unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")
    log_path = Path("path to save log") / unique_id # 日志文件的保存路径
    log_path.mkdir(exist_ok=True, parents=True) # 创建文件
    log_file = log_path / f"{unique_id}.log"  # 添加文件扩展名，生成完整的日志文件路径
    logger = set_file_handler(log_file_path=log_file)  # 使用完整的日志文件路径
    return logger
