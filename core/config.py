"""运行配置：全部来自环境变量或项目根目录的 .env。

替代原项目的 ``core/config.py``：去掉 pydantic-settings、Windows 打包路径
探测、加密凭证库与 SQLite，只保留 RAG 链路真正读取的字段，字段名与默认值
保持与原项目一致，因此移植过来的模块无需改动读取方式。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _load_env_file(path: Path) -> None:
    """加载 .env；已存在的进程环境变量优先级更高。"""
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # 未装 python-dotenv 时的最小兜底
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return
    load_dotenv(dotenv_path=path, override=False)


_load_env_file(ENV_FILE)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """RAG 链路配置快照。

    所有字段在进程启动时读取一次；修改 ``.env`` 后需重启应用。
    """

    def __init__(self) -> None:
        # 大模型（DeepSeek，兼容 OpenAI 协议）
        self.LLM_PROVIDER = _env_str("LLM_PROVIDER", "deepseek")
        self.LLM_MODEL = _env_str("LLM_MODEL", "deepseek-chat")
        self.LLM_BASE_URL = _env_str("LLM_BASE_URL", "https://api.deepseek.com/v1")
        # 演示项目沿用 DEEPSEEK_API_KEY，同时兼容原项目的 LLM_API_KEY
        self.LLM_API_KEY = _env_str(
            "LLM_API_KEY", _env_str("DEEPSEEK_API_KEY", "")
        )
        self.LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 180)
        self.LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 3)
        self.LLM_THINKING_ENABLED = _env_bool("LLM_THINKING_ENABLED", False)
        self.LLM_REASONING_EFFORT = _env_str("LLM_REASONING_EFFORT", "high")
        self.LLM_UI_TIMEOUT = _env_int("LLM_UI_TIMEOUT", 60)
        self.LLM_UI_MAX_RETRIES = _env_int("LLM_UI_MAX_RETRIES", 0)
        self.LLM_MAX_OUTPUT_TOKENS = _env_int("LLM_MAX_OUTPUT_TOKENS", 4096)

        # 嵌入模型（阿里云 DashScope / text-embedding-v4，1024 维）
        self.EMBEDDING_PROVIDER = _env_str("EMBEDDING_PROVIDER", "dashscope")
        self.EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "text-embedding-v4")
        self.EMBEDDING_BASE_URL = _env_str(
            "EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.EMBEDDING_API_KEY = _env_str("EMBEDDING_API_KEY", "")
        self.EMBEDDING_DIMENSION = _env_int("EMBEDDING_DIMENSION", 1024)
        self.EMBEDDING_QUERY_TIMEOUT = _env_int("EMBEDDING_QUERY_TIMEOUT", 20)
        self.EMBEDDING_QUERY_MAX_RETRIES = _env_int("EMBEDDING_QUERY_MAX_RETRIES", 0)

        # 向量库与论文知识库
        self.CHROMA_PERSIST_DIR = _env_str("CHROMA_PERSIST_DIR", "data/chroma_db")
        self.PAPERS_DIR = _env_str("PAPERS_DIR", "data/papers")
        self.PAPER_COLLECTION_NAME = _env_str("PAPER_COLLECTION_NAME", "plant_papers")
        self.PAPER_PILOT_COLLECTION_NAME = _env_str(
            "PAPER_PILOT_COLLECTION_NAME", "plant_papers_pilot_drought_v1"
        )
        self.PAPER_RAG_ACTIVE_COLLECTION_NAME = _env_str(
            "PAPER_RAG_ACTIVE_COLLECTION_NAME", "plant_papers_pilot_full1979_v1"
        )
        self.PAPER_CHUNK_SIZE = _env_int("PAPER_CHUNK_SIZE", 900)
        self.PAPER_CHUNK_OVERLAP = _env_int("PAPER_CHUNK_OVERLAP", 120)
        self.PAPER_MIN_TEXT_LENGTH = _env_int("PAPER_MIN_TEXT_LENGTH", 300)

        # 全量论文 RAG（混合检索 + 重排）
        assets = "data/papers/scale_all_fulltext"
        supplemental = "data/papers/data_analysis_evidence_index/vector_assets"
        self.PAPER_RAG_SCALABLE_ENABLED = _env_bool("PAPER_RAG_SCALABLE_ENABLED", True)
        self.PAPER_ENTITY_INDEX_PATH = _env_str(
            "PAPER_ENTITY_INDEX_PATH", f"{assets}/retrieval_assets/paper_entities.jsonl"
        )
        self.PAPER_ENTITY_ALIAS_PATH = _env_str(
            "PAPER_ENTITY_ALIAS_PATH", f"{assets}/retrieval_assets/entity_aliases.jsonl"
        )
        self.PAPER_CHUNKS_PATH = _env_str(
            "PAPER_CHUNKS_PATH", f"{assets}/retrieval_assets/bm25_corpus.jsonl"
        )
        self.PAPER_SUPPLEMENTAL_ENTITY_INDEX_PATH = _env_str(
            "PAPER_SUPPLEMENTAL_ENTITY_INDEX_PATH", f"{supplemental}/paper_entities.jsonl"
        )
        self.PAPER_SUPPLEMENTAL_CHUNKS_PATH = _env_str(
            "PAPER_SUPPLEMENTAL_CHUNKS_PATH", f"{supplemental}/bm25_corpus.jsonl"
        )
        self.PAPER_CITATION_CATALOG_PATH = _env_str(
            "PAPER_CITATION_CATALOG_PATH", f"{assets}/selection/proposed_all_fulltext.jsonl"
        )
        self.PAPER_EXPERIMENT_FACT_CARDS_PATH = _env_str(
            "PAPER_EXPERIMENT_FACT_CARDS_PATH", f"{assets}/fact_cards/fact_cards.jsonl"
        )
        self.PAPER_RESEARCH_TAG_INDEX_PATH = _env_str(
            "PAPER_RESEARCH_TAG_INDEX_PATH",
            f"{assets}/retrieval_assets/paper_research_tags.jsonl",
        )
        self.DATA_ANALYSIS_EVIDENCE_INDEX_PATH = _env_str(
            "DATA_ANALYSIS_EVIDENCE_INDEX_PATH",
            "data_analysis_evidence_index/papers_master.jsonl",
        )

        # 重排与召回预算
        self.PAPER_RERANK_MODEL = _env_str("PAPER_RERANK_MODEL", "qwen3-rerank")
        self.PAPER_RAG_INITIAL_CHUNKS = _env_int("PAPER_RAG_INITIAL_CHUNKS", 10)
        self.PAPER_RAG_FALLBACK_CHUNKS = _env_int("PAPER_RAG_FALLBACK_CHUNKS", 20)
        self.PAPER_RAG_LOCKED_PAPERS = _env_int("PAPER_RAG_LOCKED_PAPERS", 3)
        self.PAPER_RAG_PAPER_CANDIDATES = _env_int("PAPER_RAG_PAPER_CANDIDATES", 20)
        self.PAPER_RAG_PAPER_RERANK_WEIGHT = _env_float(
            "PAPER_RAG_PAPER_RERANK_WEIGHT", 0.2
        )
        self.PAPER_RAG_PAPER_RRF_CONSTANT = _env_int("PAPER_RAG_PAPER_RRF_CONSTANT", 30)
        self.PAPER_RAG_PAPER_LOCK_MIN_GAP = _env_float(
            "PAPER_RAG_PAPER_LOCK_MIN_GAP", 0.05
        )
        self.PAPER_RAG_RERANK_COMPARISON_PAPERS = _env_bool(
            "PAPER_RAG_RERANK_COMPARISON_PAPERS", True
        )
        self.PAPER_RECOMMENDATION_REASON_LLM_ENABLED = _env_bool(
            "PAPER_RECOMMENDATION_REASON_LLM_ENABLED", False
        )
        self.PAPER_RERANK_UI_TIMEOUT_SECONDS = _env_float(
            "PAPER_RERANK_UI_TIMEOUT_SECONDS", 30.0
        )
        self.PAPER_RERANK_UI_ATTEMPTS = _env_int("PAPER_RERANK_UI_ATTEMPTS", 1)

        # 上下文预算与应用参数
        self.RAG_MAX_CONTEXT_CHARS = _env_int("RAG_MAX_CONTEXT_CHARS", 24000)
        self.RAG_MAX_HISTORY_CHARS = _env_int("RAG_MAX_HISTORY_CHARS", 12000)
        self.RAG_MAX_REPLY_CHARS = _env_int("RAG_MAX_REPLY_CHARS", 16000)
        self.DEFAULT_DEVICE = _env_str("DEMO_DEVICE_NAME", "Device-A")
        self.MAX_CSV_SIZE_MB = _env_int("MAX_CSV_SIZE_MB", 20)
        self.LOG_LEVEL = _env_str("LOG_LEVEL", "INFO")

    # ── 路径 ──────────────────────────────────────

    @property
    def install_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def runtime_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def bundle_root(self) -> Path:
        """资源根目录（等价于项目根）。"""
        return PROJECT_ROOT

    @property
    def devices_dir(self) -> Path:
        return PROJECT_ROOT / "devices"

    @property
    def data_dir(self) -> Path:
        path = PROJECT_ROOT / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def papers_dir(self) -> Path:
        return self.resolve_path(self.PAPERS_DIR)

    @property
    def data_analysis_evidence_index_path(self) -> Path:
        candidates = (
            PROJECT_ROOT / "data" / "papers" / "data_analysis_evidence_index"
            / "papers_master_unified.jsonl",
            PROJECT_ROOT / "data" / "papers" / "data_analysis_evidence_index"
            / "papers_master_refined.jsonl",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        configured = Path(self.DATA_ANALYSIS_EVIDENCE_INDEX_PATH)
        if configured.is_absolute():
            return configured
        return (self.papers_dir / configured).resolve()

    @property
    def env_file(self) -> Path | None:
        return ENV_FILE if ENV_FILE.is_file() else None

    def resolve_path(self, relative_path: Union[str, "os.PathLike[str]"]) -> Path:
        """把配置里的相对路径解析到项目根目录下。"""
        path = Path(relative_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()


settings = Settings()


def get_llm_api_key() -> str:
    """对话模型密钥；缺失时由调用方给出可读报错。"""
    return (settings.LLM_API_KEY or "").strip()


def get_embedding_api_key() -> str:
    """嵌入模型密钥（DashScope）。"""
    return (settings.EMBEDDING_API_KEY or "").strip()


def get_credential_source(name: str) -> str:
    """凭证来源描述，供界面展示与排障。"""
    if name == "LLM_API_KEY" and get_llm_api_key():
        return "env" if os.getenv("LLM_API_KEY") else ".env(DEEPSEEK_API_KEY)"
    if name == "EMBEDDING_API_KEY" and get_embedding_api_key():
        return ".env(EMBEDDING_API_KEY)"
    return "missing"
