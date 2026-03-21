import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging() -> logging.Logger:
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"bot_log_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger('BotLogger')
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=1048576, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger