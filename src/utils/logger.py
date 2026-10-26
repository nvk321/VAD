"""Structured logging configuration using Python logging and Rich formatting."""

import logging
from typing import Optional
from rich.logging import RichHandler


def setup_logger(
    name: str = "VAD",
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """Configure a structured logger with Rich console formatting.

    Args:
        name: Name of the logger.
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional file path to mirror logs to disk.

    Returns:
        logging.Logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False
        )
        console_handler.setLevel(numeric_level)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
