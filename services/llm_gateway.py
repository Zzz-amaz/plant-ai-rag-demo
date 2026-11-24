"""模型网关：凭证统一来自 ``.env``，不提供任何演示降级。

缺少 Key 时直接抛 :class:`MissingCredentialsError`，由页面渲染
「请配置 Key」的错误提示；有 Key 时返回真实的 ChatOpenAI 客户端。
"""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI

from agents.llm_factory import create_llm as _create_real_llm
from core.config import get_embedding_api_key, get_llm_api_key, settings

MISSING_KEY_MESSAGE = (
    "请配置 Key：在项目根目录的 .env 中填写 `LLM_API_KEY` 与 `EMBEDDING_API_KEY`"
    "（可参考 .env.example），保存后重启应用。"
)


class MissingCredentialsError(RuntimeError):
    """缺少模型或嵌入凭证。"""


def missing_credentials() -> list[str]:
    """返回缺失的配置项名称，供界面逐条提示。"""
    missing: list[str] = []
    if not get_llm_api_key():
        missing.append("LLM_API_KEY")
    if not get_embedding_api_key():
        missing.append("EMBEDDING_API_KEY")
    return missing


def credentials_ready() -> bool:
    return not missing_credentials()


def require_credentials() -> None:
    """缺少凭证时抛出可读异常。"""
    missing = missing_credentials()
    if missing:
        raise MissingCredentialsError(
            f"{MISSING_KEY_MESSAGE}（缺少：{', '.join(missing)}）"
        )


def model_label() -> str:
    """当前模型标识，用于侧边栏与首页徽标。"""
    return settings.LLM_MODEL


def embedding_label() -> str:
    return settings.EMBEDDING_MODEL


def create_llm(streaming: bool = False) -> Optional[ChatOpenAI]:
    """构建真实模型客户端；凭证缺失时抛 :class:`MissingCredentialsError`。"""
    require_credentials()
    return _create_real_llm(streaming=streaming, interactive=streaming)
