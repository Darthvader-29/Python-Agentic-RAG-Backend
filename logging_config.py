"""Structured logging (structlog). Call configure_logging() once at startup.
Per module: logger = structlog.get_logger(__name__)."""

import logging
import sys

import structlog


def configure_logging(json_logs: bool | None = None) -> None:
    if json_logs is None:
        from config import settings

        json_logs = settings.LOG_JSON
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
