"""Small logging helpers for PyFLI."""

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
    Run the format message routine.

    Parameters
    ----------
    msg : Any
        Message text emitted through the PyFLI logger.
    *args : Any
        Additional positional values accepted by the routine.
    **kwargs : Any
        Additional keyword options forwarded to the underlying implementation.

    Returns
    -------
    str
        String path, label, or message produced by format message.
    """
    if not args and not kwargs:
        return str(msg)
    return str(msg).format(*args, **kwargs)


def _log(level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Run the log routine.

    Parameters
    ----------
    level : int
        Logging severity level.
    msg : Any
        Message text emitted through the PyFLI logger.
    *args : Any
        Additional positional values accepted by the routine.
    **kwargs : Any
        Additional keyword options forwarded to the underlying implementation.

    Returns
    -------
    None
        No object is returned; the function perform log.
    """
    logger.log(level, _format_message(msg, *args, **kwargs))


def critical(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Run the critical routine.

    Parameters
    ----------
    msg : Any
        Message text emitted through the PyFLI logger.
    *args : Any
        Additional positional values accepted by the routine.
    **kwargs : Any
        Additional keyword options forwarded to the underlying implementation.

    Returns
    -------
    None
        No object is returned; the function perform critical.
    """
    _log(_logging.CRITICAL, msg, *args, **kwargs)


def debug(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Run the debug routine.

    Parameters
    ----------
    msg : Any
        Message text emitted through the PyFLI logger.
    *args : Any
        Additional positional values accepted by the routine.
    **kwargs : Any
        Additional keyword options forwarded to the underlying implementation.

    Returns
    -------
    None
        No object is returned; the function perform debug.
    """
    _log(_logging.DEBUG, msg, *args, **kwargs)


def error(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Run the error routine.

    Parameters
    ----------
    msg : Any
        Message text emitted through the PyFLI logger.
    *args : Any
        Additional positional values accepted by the routine.
    **kwargs : Any
        Additional keyword options forwarded to the underlying implementation.

    Returns
    -------
    None
        No object is returned; the function perform error.
    """
    _log(_logging.ERROR, msg, *args, **kwargs)


def exception(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Run the exception routine.

    Parameters
    ----------
    msg : Any
        Message text emitted through the PyFLI logger.
    *args : Any
        Additional positional values accepted by the routine.
    **kwargs : Any
        Additional keyword options forwarded to the underlying implementation.

    Returns
    -------
    None
        No object is returned; the function perform exception.
    """
    logger.exception(_format_message(msg, *args, **kwargs))


def info(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Run the info routine.

    Parameters
    ----------
    msg : Any
        Message text emitted through the PyFLI logger.
    *args : Any
        Additional positional values accepted by the routine.
    **kwargs : Any
        Additional keyword options forwarded to the underlying implementation.

    Returns
    -------
    None
        No object is returned; the function perform info.
    """
    _log(_logging.INFO, msg, *args, **kwargs)


def log(level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Run the log routine.

    Parameters
    ----------
    level : int
        Logging severity level.
    msg : Any
        Message text emitted through the PyFLI logger.
    *args : Any
        Additional positional values accepted by the routine.
    **kwargs : Any
        Additional keyword options forwarded to the underlying implementation.

    Returns
    -------
    None
        No object is returned; the function perform log.
    """
    _log(level, msg, *args, **kwargs)


def warning(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Run the warning routine.

    Parameters
    ----------
    msg : Any
        Message text emitted through the PyFLI logger.
    *args : Any
        Additional positional values accepted by the routine.
    **kwargs : Any
        Additional keyword options forwarded to the underlying implementation.

    Returns
    -------
    None
        No object is returned; the function perform warning.
    """
    _log(_logging.WARNING, msg, *args, **kwargs)


@lru_cache(None)
def warn_once(msg: Any, *args: Any, **kwargs: Any) -> None:
    """
    Run the warn once routine.

    Parameters
    ----------
    msg : Any
        Message text emitted through the PyFLI logger.
    *args : Any
        Additional positional values accepted by the routine.
    **kwargs : Any
        Additional keyword options forwarded to the underlying implementation.

    Returns
    -------
    None
        No object is returned; the function perform warn once.
    """
    warning(msg, *args, **kwargs)


def format_duration(seconds: float) -> str:
    """
    Format an elapsed time in seconds, minutes, or hours.

    Parameters
    ----------
    seconds : float
        Elapsed time in seconds.

    Returns
    -------
    str
        String path, label, or message produced by format duration.
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"

    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} minutes"

    hours = minutes / 60
    return f"{hours:.2f} hours"
