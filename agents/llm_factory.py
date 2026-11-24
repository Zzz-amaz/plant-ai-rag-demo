r"""LLM 工厂 —— 抽象工厂模式，对齐 langchain 社区最佳实践。

参考项目: E:\大模型\langchain\AI大模型RAG与智能体开发_Agent项目\model\factory.py
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI

from core.config import get_llm_api_key, settings


class BaseModelFactory(ABC):
    """模型工厂抽象基类。"""

    @abstractmethod
    def create(self) -> ChatOpenAI:
        """创建模型实例。"""
        pass


class ChatModelFactory(BaseModelFactory):
    """LLM 对话模型工厂。

    创建兼容 OpenAI API 的 ChatModel（DeepSeek / 千问 / 等）。
    """

    def __init__(self, streaming: bool = True, interactive: bool = False):
        self.streaming = streaming
        self.interactive = interactive

    def create(self) -> ChatOpenAI:
        api_key = get_llm_api_key()
        if not api_key:
            env_location = str(settings.env_file) if settings.env_file else "未找到 .env"
            raise RuntimeError(
                "未检测到 LLM_API_KEY。请在项目根目录的 .env 中配置 "
                "LLM_API_KEY（或 DEEPSEEK_API_KEY），参考 .env.example 后重启应用。"
                f" 当前 .env 来源：{env_location}"
            )

        return _build_cached_llm(
            api_key=api_key,
            model=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL,
            timeout=(settings.LLM_UI_TIMEOUT if (self.streaming or self.interactive) else settings.LLM_TIMEOUT),
            max_retries=(settings.LLM_UI_MAX_RETRIES if (self.streaming or self.interactive) else settings.LLM_MAX_RETRIES),
            streaming=self.streaming,
            thinking_enabled=settings.LLM_THINKING_ENABLED,
            reasoning_effort=settings.LLM_REASONING_EFFORT,
        )


@lru_cache(maxsize=8)
def _build_cached_llm(
    *,
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    max_retries: int,
    streaming: bool,
    thinking_enabled: bool,
    reasoning_effort: str,
) -> ChatOpenAI:
    """Reuse HTTP clients between turns while naturally invalidating on key change."""
    extra_body = {"thinking": {"type": "enabled" if thinking_enabled else "disabled"}}
    return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.3,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            timeout=timeout,
            max_retries=max_retries,
            streaming=streaming,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
        )


# 模块级单例（非流式，供内部调用如摘要生成、Query Rewriting 等）
_default_llm: Optional[ChatOpenAI] = None


def create_llm(streaming: bool = True, *, interactive: bool = False) -> ChatOpenAI:
    """创建 LLM 实例（便捷函数，保持向后兼容）。

    Args:
        streaming: 是否启用流式输出

    Returns:
        配置好的 ChatOpenAI 实例
    """
    return ChatModelFactory(streaming=streaming, interactive=interactive).create()
