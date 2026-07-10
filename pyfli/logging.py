"""Small logging helpers for PyFLI."""

import logging as _logging
from functools import lru_cache

LOGGER_NAME = "pyfli"

logger = _logging.getLogger(LOGGER_NAME)


def configure_logging(level: int = _logging.INFO) -> _logging.Logger:
    """Configure PyFLI logging without overriding user logging handlers."""
    if not _logging.getLogger().handlers:
        _logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")

    logger.setLevel(level)
    return logger


def _format_message(msg, *args, **kwargs) -> str:
    if not args and not kwargs:
        return str(msg)
    return str(msg).format(*args, **kwargs)


def _log(level: int, msg, *args, **kwargs) -> None:
    logger.log(level, _format_message(msg, *args, **kwargs))


def critical(msg, *args, **kwargs) -> None:
    _log(_logging.CRITICAL, msg, *args, **kwargs)


def debug(msg, *args, **kwargs) -> None:
    _log(_logging.DEBUG, msg, *args, **kwargs)


def error(msg, *args, **kwargs) -> None:
    _log(_logging.ERROR, msg, *args, **kwargs)


def exception(msg, *args, **kwargs) -> None:
    logger.exception(_format_message(msg, *args, **kwargs))


def info(msg, *args, **kwargs) -> None:
    _log(_logging.INFO, msg, *args, **kwargs)


def log(level: int, msg, *args, **kwargs) -> None:
    _log(level, msg, *args, **kwargs)


def warning(msg, *args, **kwargs) -> None:
    _log(_logging.WARNING, msg, *args, **kwargs)


@lru_cache(None)
def warn_once(msg, *args, **kwargs) -> None:
    warning(msg, *args, **kwargs)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} seconds"

    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} minutes"

    hours = minutes / 60
    return f"{hours:.2f} hours"
