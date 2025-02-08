import os
import secrets
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, FileHandler, Logger
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


LOG_LEVELS = {
    'debug': DEBUG,
    'info': INFO,
    'warning': WARNING,
    'error': ERROR,
    'critical': CRITICAL,
}


def test_logger_file_handlers(log_dir: Path):
    os.environ['NEXTPREDCO_LOG_DIR'] = str(log_dir)

    from nextpredco.core._logger import logger

    original_messages = {
        'debug': 'This is a debug message.',
        'info': 'This is an info message.',
        'warning': 'This is a warning message.',
        'error': 'This is an error message.',
        'critical': 'This is a critical message.',
    }

    # Generate a list of 10 random key-value pairs from original_messages
    random_messages = {}
    for _ in range(20):
        key = secrets.choice(list(original_messages.keys()))
        value = original_messages[key]
        random_messages[key] = value

    # Test the logger with the original messages
    for level, message in random_messages.items():
        getattr(logger, level)(message)
        check_message_in_file(logger, level, message)


def check_message_in_file(logger: Logger, level: str, message: str):
    for handler in logger.handlers:
        if isinstance(handler, FileHandler | TimedRotatingFileHandler):
            with Path(handler.baseFilename).open() as file:
                if handler.level < LOG_LEVELS[level]:
                    lines = file.readlines()
                    if message not in lines[-1]:
                        error = (
                            f"Expected '{message}' "
                            'to be in the last line of the log file.'
                        )
                        raise AssertionError(error)
