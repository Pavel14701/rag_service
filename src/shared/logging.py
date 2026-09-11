"""Logging configuration.

Structured logging via structlog. When ``json_output`` is enabled every
log record is rendered as a single JSON line; otherwise a colored console
renderer is used. ``merge_contextvars`` folds bound context (``request_id``,
``queue``) set via ``structlog.contextvars`` into every record.
"""

import logging
import sys

import structlog


def configure_logging(level: str = 'INFO', json_output: bool = True) -> None:
    """Configure structlog and stdlib logging."""
    logging.basicConfig(
        level=level.upper(), stream=sys.stdout, format='%(message)s'
    )

    renderer: structlog.typing.Processor
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt='iso'),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
