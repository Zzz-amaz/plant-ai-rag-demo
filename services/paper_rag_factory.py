"""Construct the disabled-by-default scalable paper RAG grey path."""

from __future__ import annotations

from agents.llm_factory import create_llm
from core.config import get_embedding_api_key, settings
from evaluation.qwen_rerank import QwenRerankClient
from rag.chroma_paper_search import ChromaPaperSearch
from rag.entity_alias_catalog import EntityAliasCatalog
from rag.local_chunk_search import JsonlPaperChunkSearch
from rag.paper_entities import PaperEntityIndex, read_profiles
from rag.provider_adapters import ConstraintAwareQwenPaperRanker, DeepSeekCoverageAnswerer, QwenChunkReranker
from rag.paper_rank_fusion import ConstraintAwarePaperRanker
from rag.scalable_retrieval import ScalablePaperRetriever
from rag.vector_store import get_paper_vector_store

from .paper_rag_pipeline import PaperRagPipeline


def build_scalable_paper_rag_pipeline(*, interactive: bool = False) -> PaperRagPipeline:
    """Build on demand; construction alone makes no external API calls."""

    profiles_path = settings.resolve_path(settings.PAPER_ENTITY_INDEX_PATH)
    alias_path = settings.resolve_path(settings.PAPER_ENTITY_ALIAS_PATH)
    chunks_path = settings.resolve_path(settings.PAPER_CHUNKS_PATH)
    supplemental_profiles_path = settings.resolve_path(settings.PAPER_SUPPLEMENTAL_ENTITY_INDEX_PATH)
    profiles = list(read_profiles(profiles_path))
    if supplemental_profiles_path.is_file():
        # Imported PDF profiles deliberately supplement rather than replace
        # the frozen full corpus.  A paper ID collision keeps the canonical
        # full-corpus profile.
        known_ids = {profile.paper_id for profile in profiles}
        profiles.extend(
            profile for profile in read_profiles(supplemental_profiles_path)
            if profile.paper_id not in known_ids
        )
    entity_index = PaperEntityIndex(profiles)
    alias_catalog = EntityAliasCatalog.read_jsonl(alias_path)
    vector_store = get_paper_vector_store()
    paper_search = ChromaPaperSearch(vector_store, entity_index)
    supplemental_chunks_path = settings.resolve_path(settings.PAPER_SUPPLEMENTAL_CHUNKS_PATH)
    chunk_search = JsonlPaperChunkSearch(
        [chunks_path, supplemental_chunks_path]
        if supplemental_chunks_path.is_file()
        else chunks_path
    )
    rerank_client = QwenRerankClient(
        get_embedding_api_key(),
        model=settings.PAPER_RERANK_MODEL,
        timeout_seconds=(settings.PAPER_RERANK_UI_TIMEOUT_SECONDS if interactive else 120.0),
        default_attempts=(settings.PAPER_RERANK_UI_ATTEMPTS if interactive else 3),
        # The SDK path does not expose a reliable request timeout.  Use the
        # HTTP adapter for the interactive path so its budget is enforced.
        force_http=interactive,
    )
    retriever = ScalablePaperRetriever(
        entity_index,
        chunk_search,
        paper_search=paper_search,
        chunk_reranker=QwenChunkReranker(rerank_client),
        locked_paper_limit=settings.PAPER_RAG_LOCKED_PAPERS,
        initial_chunks_per_paper=settings.PAPER_RAG_INITIAL_CHUNKS,
        fallback_chunks_per_paper=settings.PAPER_RAG_FALLBACK_CHUNKS,
        alias_catalog=alias_catalog,
        paper_reranker=ConstraintAwareQwenPaperRanker(
            rerank_client,
            ConstraintAwarePaperRanker(
                alias_catalog,
                rerank_weight=settings.PAPER_RAG_PAPER_RERANK_WEIGHT,
                rank_constant=settings.PAPER_RAG_PAPER_RRF_CONSTANT,
            ),
        ),
        rerank_comparison_papers=settings.PAPER_RAG_RERANK_COMPARISON_PAPERS,
        paper_candidate_limit=settings.PAPER_RAG_PAPER_CANDIDATES,
        paper_lock_min_score_gap=settings.PAPER_RAG_PAPER_LOCK_MIN_GAP,
    )
    answerer = DeepSeekCoverageAnswerer(
        create_llm(streaming=False, interactive=interactive)
    )
    return PaperRagPipeline(retriever, answerer)
