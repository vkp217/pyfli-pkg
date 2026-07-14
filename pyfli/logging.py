"""Small logging helpers for PyFLI."""

from __future__ import annotations
from typing import Any
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


def _format_message(msg: Any, *args: Any, **kwargs: Any) -> str:
    """
    Handle format message.

    Parameters
    ----------
    msg : Any
        Input value.
    *args : Any
        Input value.
    **kwargs : Any
        Input value.

    Returns
    -------
    str
        Return value.
    """
    if not args and not kwargs:
        return str(msg)
    return str(msg).format(*args, **kwargs)


def _log(level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Handle log.

    Parameters
    ----------
    level : int
        Input value.
    msg : Any
        Input value.
    *args : Any
        Input value.
    **kwargs : Any
        Input value.

    Returns
    -------
    None
        Return value.
    """
    logger.log(level, _format_message(msg, *args, **kwargs))


def critical(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Handle critical.

    Parameters
    ----------
    msg : Any
        Input value.
    *args : Any
        Input value.
    **kwargs : Any
        Input value.

    Returns
    -------
    None
        Return value.
    """
    _log(_logging.CRITICAL, msg, *args, **kwargs)


def debug(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Handle debug.

    Parameters
    ----------
    msg : Any
        Input value.
    *args : Any
        Input value.
    **kwargs : Any
        Input value.

    Returns
    -------
    None
        Return value.
    """
    _log(_logging.DEBUG, msg, *args, **kwargs)


def error(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Handle error.

    Parameters
    ----------
    msg : Any
        Input value.
    *args : Any
        Input value.
    **kwargs : Any
        Input value.

    Returns
    -------
    None
        Return value.
    """
    _log(_logging.ERROR, msg, *args, **kwargs)


def exception(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Handle exception.

    Parameters
    ----------
    msg : Any
        Input value.
    *args : Any
        Input value.
    **kwargs : Any
        Input value.

    Returns
    -------
    None
        Return value.
    """
    logger.exception(_format_message(msg, *args, **kwargs))


def info(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Handle info.

    Parameters
    ----------
    msg : Any
        Input value.
    *args : Any
        Input value.
    **kwargs : Any
        Input value.

    Returns
    -------
    None
        Return value.
    """
    _log(_logging.INFO, msg, *args, **kwargs)


def log(level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Handle log.

    Parameters
    ----------
    level : int
        Input value.
    msg : Any
        Input value.
    *args : Any
        Input value.
    **kwargs : Any
        Input value.

    Returns
    -------
    None
        Return value.
    """
    _log(level, msg, *args, **kwargs)


def warning(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Handle warning.

    Parameters
    ----------
    msg : Any
        Input value.
    *args : Any
        Input value.
    **kwargs : Any
        Input value.

    Returns
    -------
    None
        Return value.
    """
    _log(_logging.WARNING, msg, *args, **kwargs)


@lru_cache(None)
def warn_once(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Handle warn once.

    Parameters
    ----------
    msg : Any
        Input value.
    *args : Any
        Input value.
    **kwargs : Any
        Input value.

    Returns
    -------
    None
        Return value.
    """
    warning(msg, *args, **kwargs)


def format_duration(seconds: float) -> str:
    """
    Handle format duration.

    Parameters
    ----------
    seconds : float
        Input value.

    Returns
    -------
    str
        Return value.
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"

    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} minutes"

    hours = minutes / 60
    return f"{hours:.2f} hours"
