"""向量库管理器 —— ChromaDB 封装，使用千问 DashScope Embedding API。

支持多设备知识库隔离（每个设备一个 Collection）。
提供索引、查询、删除等操作。
"""

from __future__ import annotations

import sys
import shutil
import hashlib
from threading import RLock
from pathlib import Path

import requests
import numpy as np
import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from core.logging import logger

from core.config import get_embedding_api_key, settings


class DashScopeEmbeddingFn(EmbeddingFunction):
    """ChromaDB 兼容的千问 DashScope Embedding 适配器。

    直接通过 HTTP 请求调用 DashScope OpenAI 兼容接口，
    避免 langchain_openai 的 SDK 参数格式不兼容问题。
    """

    def __init__(self):
        # 密钥在每次请求前动态读取，支持在页面中保存后立即生效。
        self._model = settings.EMBEDDING_MODEL
        self._base_url = settings.EMBEDDING_BASE_URL.rstrip("/")
        self._url = f"{self._base_url}/embeddings"
        self._session = requests.Session()
        logger.info(f"初始化 Embedding: {self._model} @ {self._base_url}")

    def __call__(self, input: list[str]) -> list[list[float]]:
        """ChromaDB 要求: (self, input: list[str]) -> list[list[float]]"""
        embeddings = self._embed(input)
        return [np.array(e, dtype=np.float32).tolist() for e in embeddings]

    def _post_with_retry(
        self,
        batch: list[str],
        *,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> dict:
        """带指数退避 + 抖动重试的 Embedding API 调用。"""
        import time as _time
        import random as _random

        base_delay = 1.0  # 基础延迟秒数

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                api_key = get_embedding_api_key()
                if not api_key:
                    raise RuntimeError(
                        "未配置 EMBEDDING_API_KEY。请通过首页右下角的 API 密钥设置按钮保存密钥。"
                    )

                resp = self._session.post(
                    self._url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self._model, "input": batch},
                    timeout=timeout,
                )

                if resp.status_code == 200:
                    return resp.json()

                # 429 或 5xx 才值得重试，4xx（如 401 认证失败）直接报错
                if resp.status_code in (429, 502, 503, 504):
                    last_error = RuntimeError(
                        f"Embedding API {resp.status_code}（第{attempt + 1}次尝试）: {resp.text[:200]}"
                    )
                else:
                    raise RuntimeError(
                        f"Embedding API 错误 ({resp.status_code}): {resp.text[:300]}"
                    )

            except requests.exceptions.Timeout as e:
                last_error = e
            except requests.exceptions.ConnectionError as e:
                last_error = e

            if attempt < max_retries:
                # 指数退避 + 随机抖动
                delay = base_delay * (2 ** attempt) + _random.uniform(0, 0.5)
                logger.warning(
                    f"Embedding API 调用失败，{delay:.1f}s 后重试（{attempt + 2}/{max_retries + 1}）"
                )
                _time.sleep(delay)

        raise last_error  # type: ignore[misc]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """分批调用 DashScope Embedding API（带自动重试）。"""
        batch_size = 10  # DashScope text-embedding-v4 单次限制
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            data = self._post_with_retry(batch)
            for item in data["data"]:
                all_embeddings.append(item["embedding"])

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed one interactive query with a short, non-retrying budget."""

        data = self._post_with_retry(
            [text],
            timeout=settings.EMBEDDING_QUERY_TIMEOUT,
            max_retries=settings.EMBEDDING_QUERY_MAX_RETRIES,
        )
        rows = data.get("data") or []
        if not rows or not rows[0].get("embedding"):
            raise RuntimeError("Embedding query returned no vector")
        return np.asarray(rows[0]["embedding"], dtype=np.float32).tolist()


# 全局单例
_client: chromadb.PersistentClient | None = None  # type: ignore[valid-type]
_embedding_fn: DashScopeEmbeddingFn | None = None
_vector_stores: dict[str, "VectorStore"] = {}
_runtime_lock = RLock()


def get_embedding_function() -> DashScopeEmbeddingFn:
    """获取 ChromaDB 兼容的嵌入函数单例。"""
    global _embedding_fn
    with _runtime_lock:
        if _embedding_fn is None:
            _embedding_fn = DashScopeEmbeddingFn()
        return _embedding_fn


def reset_vector_store_runtime() -> None:
    """Reset cached wrappers after credentials are changed from the UI."""
    global _embedding_fn, _paper_vector_store
    with _runtime_lock:
        _embedding_fn = None
        _vector_stores.clear()
        _paper_vector_store = None


def _seed_chromadb_from_bundle(target_dir: Path) -> None:
    """首次运行打包版时复制可选的预构建 ChromaDB。

    查找顺序：
    1. EXE 同级 ``seed_data/chroma_db``（便于单独分发大知识库）；
    2. PyInstaller ``_internal/data/chroma_db``（full 构建模式）。
    """
    candidates = [
        settings.install_root / "seed_data" / "chroma_db",
        settings.bundle_root / "data" / "chroma_db",
    ]
    if target_dir.exists():
        return

    for source_dir in candidates:
        if not source_dir.is_dir():
            continue
        logger.info(f"[种子数据] 正在复制知识库索引: {source_dir} -> {target_dir}")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(source_dir), str(target_dir))
        logger.info("[种子数据] 知识库索引复制完成")
        return


def get_chroma_client() -> chromadb.PersistentClient:
    """获取 ChromaDB 客户端单例。"""
    global _client
    with _runtime_lock:
        if _client is None:
            persist_dir = settings.resolve_path(settings.CHROMA_PERSIST_DIR)
            if getattr(sys, 'frozen', False):
                _seed_chromadb_from_bundle(persist_dir)
            logger.info(f"初始化 ChromaDB: {persist_dir}")
            _client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return _client


class VectorStore:
    """单个设备的向量库。"""

    def __init__(self, device_name: str | None = None,
                 collection_name: str | None = None):
        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = f"plant_{device_name}"
        self.device_name = device_name or collection_name or "unknown"

        client = get_chroma_client()

        try:
            # Collections persisted by different Chroma releases can carry an
            # incompatible embedding-function signature.  We always submit
            # explicit vectors below, so opening without one is safe and
            # prevents a false conflict from recreating an existing corpus.
            self.collection = client.get_collection(name=self.collection_name)
        except Exception:
            self.collection = client.create_collection(
                name=self.collection_name,
                metadata={"device": self.device_name},
            )

    @property
    def chunk_count(self) -> int:
        with _runtime_lock:
            return self.collection.count()

    def index_documents(
        self,
        documents: list[Document],
        force_reindex: bool = False,
        stable_ids: bool = False,
    ) -> int:
        """将文档块索引到向量库。"""
        if not documents:
            return 0

        if force_reindex and self.collection.count() > 0:
            client = get_chroma_client()
            client.delete_collection(name=self.collection_name)
            self.collection = client.create_collection(
                name=self.collection_name,
                metadata={"device": self.device_name},
            )
            logger.info(f"已清空向量库: {self.device_name}")

        if stable_ids:
            # 论文索引使用内容稳定 ID；重复执行索引时覆盖原块，而不是持续追加。
            unique_documents: dict[str, Document] = {}
            for doc in documents:
                paper_id = str(doc.metadata.get("paper_id", "")).strip().lower()
                identity = f"{paper_id}\n{doc.page_content.strip()}"
                chunk_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                unique_documents[f"paper_{chunk_hash}"] = doc
            ids = list(unique_documents)
            documents = list(unique_documents.values())
        else:
            ids = [f"{self.device_name}_{i}" for i in range(
                self.collection.count(),
                self.collection.count() + len(documents)
            )]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # ChromaDB 单次写入上限 5461，分批次写入
        BATCH_LIMIT = 5000
        total = len(documents)
        for start in range(0, total, BATCH_LIMIT):
            end = min(start + BATCH_LIMIT, total)
            batch_embeddings = get_embedding_function()(texts[start:end])
            if stable_ids:
                self.collection.upsert(
                    ids=ids[start:end],
                    documents=texts[start:end],
                    metadatas=metadatas[start:end],
                    embeddings=batch_embeddings,
                )
            else:
                self.collection.add(
                    ids=ids[start:end],
                    documents=texts[start:end],
                    metadatas=metadatas[start:end],
                    embeddings=batch_embeddings,
                )
            logger.debug(f"已写入 {end}/{total} 块到 [{self.device_name}]")

        logger.info(f"已索引 {total} 个文档块到设备 [{self.device_name}]")
        return total

    def deduplicate_by_content(self) -> int:
        """删除同一论文的完全重复文本块，返回删除数量。"""
        if self.collection.count() < 2:
            return 0

        data = self.collection.get(include=["documents", "metadatas"])
        seen: dict[tuple[str, str], str] = {}
        duplicate_ids: list[str] = []
        for chunk_id, content, metadata in zip(
            data["ids"], data["documents"], data["metadatas"]
        ):
            paper_id = str((metadata or {}).get("paper_id", "")).strip().lower()
            key = (paper_id, (content or "").strip())
            previous_id = seen.get(key)
            if previous_id is None:
                seen[key] = chunk_id
            elif chunk_id.startswith("paper_"):
                # 优先保留新格式的稳定 ID，便于后续 upsert。
                duplicate_ids.append(previous_id)
                seen[key] = chunk_id
            else:
                duplicate_ids.append(chunk_id)

        if duplicate_ids:
            self.collection.delete(ids=duplicate_ids)
            logger.info(f"已清理论文库完全重复块: {len(duplicate_ids)} 个")
        return len(duplicate_ids)

    def similarity_search(
        self, query: str, k: int = 4, include_embeddings: bool = False
    ) -> list[Document]:
        """语义相似度检索。"""
        # Chroma's native backend is not safe to initialize/query concurrently
        # while another Streamlit rerun is opening the persistent collection.
        # Serialize the short native operation; the HTTP embedding request is
        # intentionally performed before entering the lock.
        with _runtime_lock:
            count = self.collection.count()
            if count == 0:
                return []

        query_embedding = get_embedding_function().embed_query(query)
        with _runtime_lock:
            count = self.collection.count()
            if count == 0:
                return []

            includes = ["documents", "metadatas", "distances"]
            if include_embeddings:
                includes.append("embeddings")

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, count),
                include=includes,
            )

        documents = []
        if results["documents"] and results["documents"][0]:
            for i, text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                doc = Document(
                    page_content=text,
                    metadata={**metadata, "score": 1 - distance},
                )
                if include_embeddings and results.get("embeddings"):
                    doc.metadata["embedding"] = results["embeddings"][0][i]
                documents.append(doc)

        return documents

    def delete_collection(self):
        """删除当前设备的向量库。"""
        client = get_chroma_client()
        try:
            client.delete_collection(name=self.collection_name)
            logger.info(f"已删除向量库: {self.device_name}")
        except Exception as e:
            logger.warning(f"删除向量库失败: {e}")


def get_vector_store(device_name: str | None = None) -> VectorStore:
    """获取设备对应的向量库（带缓存）。"""
    device_name = device_name or settings.DEFAULT_DEVICE
    with _runtime_lock:
        if device_name not in _vector_stores:
            _vector_stores[device_name] = VectorStore(device_name)
        return _vector_stores[device_name]


def index_device(device_name: str | None = None,
                 force_reindex: bool = False) -> int:
    """为指定设备构建/重建知识库（支持增量索引）。

    增量模式下通过 MD5 检测文件变化，只处理新增/修改的文件。
    force_reindex=True 时全量重建。
    """
    from .document_loader import load_device_documents
    from .text_splitter import split_documents

    device_name = device_name or settings.DEFAULT_DEVICE
    incremental = not force_reindex
    logger.info(f"开始为设备 [{device_name}] 构建知识库（{'增量' if incremental else '全量'}）...")

    store = get_vector_store(device_name)
    if incremental and store.chunk_count == 0:
        # collection 可能仍存在但内容已丢失，此时必须全量加载，不能只看 MD5。
        logger.warning(f"设备 [{device_name}] 向量库为空，执行全量重建")
        documents, is_incremental = load_device_documents(device_name, incremental=False)
    else:
        documents, is_incremental = load_device_documents(device_name, incremental=incremental)

    # 增量模式下如果 ChromaDB 被清空（如手动删除 chroma_db 目录），
    # 回退为全量重建
    if is_incremental and not documents:
        if store.chunk_count == 0:
            logger.warning(
                f"设备 [{device_name}] 向量库为空，回退为全量重建"
            )
            documents, is_incremental = load_device_documents(device_name, incremental=False)

    if not documents:
        if is_incremental:
            logger.info(f"设备 [{device_name}] 文档无变化，跳过")
        else:
            logger.warning(f"设备 [{device_name}] 没有可索引的文档")
        return 0

    chunks = split_documents(documents)
    if not chunks:
        return 0

    count = store.index_documents(
        chunks,
        # A parser/schema change must rebuild the collection; otherwise old
        # chunks from the previous layout would remain alongside new chunks.
        force_reindex=(force_reindex or not is_incremental),
        stable_ids=True,
    )
    store.deduplicate_by_content()

    logger.info(f"设备 [{device_name}] 知识库构建完成: {count} 个文档块")
    return count


def list_indexed_devices() -> list[str]:
    """列出所有已构建知识库的设备。"""
    client = get_chroma_client()
    collections = client.list_collections()
    return [
        col.name.replace("plant_", "")
        for col in collections
        if col.name.startswith("plant_")
    ]


# --- 论文知识库 ---

_paper_vector_store: VectorStore | None = None


def get_paper_vector_store(collection_name: str | None = None) -> VectorStore:
    """获取论文向量存储；灰度链路默认读取经验证的隔离全量集合。"""
    global _paper_vector_store
    active_name = collection_name or settings.PAPER_RAG_ACTIVE_COLLECTION_NAME
    with _runtime_lock:
        if _paper_vector_store is None or _paper_vector_store.collection_name != active_name:
            _paper_vector_store = VectorStore(collection_name=active_name)
        return _paper_vector_store


def index_papers(force_reindex: bool = False, limit: int | None = None,
                 offset: int = 0) -> int:
    """构建/重建论文知识库索引。

    Args:
        force_reindex: True 时清空重建，False 时增量追加
        limit: 限制索引论文数量（None = 全部）
        offset: 跳过前 N 篇 ready 论文（用于分批续传）

    Returns:
        索引的文档块数量
    """
    from .paper_loader import load_papers_from_manifest
    from .text_splitter import split_documents

    logger.info(f"开始构建论文知识库（{'全量' if force_reindex else '增量'}，offset={offset}，limit={limit}）...")

    documents = load_papers_from_manifest(limit=limit, offset=offset)
    if not documents:
        logger.warning("没有可索引的论文")
        return 0

    chunks = split_documents(documents)
    if not chunks:
        return 0

    # 索引维护始终写入正式集合，绝不意外覆盖灰度验证集合。
    store = get_paper_vector_store(settings.PAPER_COLLECTION_NAME)
    count = store.index_documents(chunks, force_reindex=force_reindex, stable_ids=True)
    store.deduplicate_by_content()

    logger.info(f"论文知识库构建完成: {count} 个文档块（collection 总量: {store.chunk_count}）")
    return count
