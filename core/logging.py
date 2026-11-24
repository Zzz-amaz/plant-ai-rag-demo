"""统一日志入口：替代原项目的 loguru，输出到标准错误流。

原模块只用到 ``info/warning/error/debug/exception`` 五个方法，
这里做一个薄包装，同时兼容 ``logger.info("x {}", v)`` 的花括号占位写法。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("plant")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
        )
        root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    root.propagate = False
    _CONFIGURED = True


class _CompatLogger:
    """loguru 风格的最小适配器。"""

    def __init__(self, name: str) -> None:
        _configure()
        self._logger = logging.getLogger(name)

    @staticmethod
    def _render(message: str, args: tuple[Any, ...]) -> str:
        if not args:
            return message
        if "{}" in message:
            try:
                return message.format(*args)
            except (IndexError, KeyError, ValueError):
                return message
        return f"{message} {' '.join(str(arg) for arg in args)}"

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(self._render(message, args))

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(self._render(message, args))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(self._render(message, args))

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(self._render(message, args))

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(self._render(message, args))


logger = _CompatLogger("plant.rag")


def get_logger(name: str) -> _CompatLogger:
    """按模块名获取日志对象。"""
    return _CompatLogger(f"plant.{name}")
