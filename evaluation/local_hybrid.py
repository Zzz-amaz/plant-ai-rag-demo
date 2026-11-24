"""Small dependency-free BM25/RRF helpers for the pilot retrieval experiments."""

from __future__ import annotations

import math
import re
from collections import Counter


DOMAIN_EXPANSIONS = {
    "菜豆": "common bean phaseolus vulgaris",
    "花生": "peanut",
    "番茄": "tomato",
    "罗勒": "basil",
    "牛至": "oregano",
    "玉米": "maize corn",
    "大豆": "soybean",
    "小麦": "wheat",
    "甘蔗": "sugarcane",
    "黑麦草": "ryegrass",
    "田间持水量": "field water capacity field capacity",
    "水势": "water potential",
    "叶绿素荧光": "chlorophyll fluorescence",
    "荧光成像": "fluorescence imaging",
    "暗适应": "dark adaptation dark-adapted",
    "光强": "light intensity irradiance",
    "低光": "low light LL",
    "高光": "high light HL",
    "水杨酸": "salicylic acid SA",
    "间歇": "intermittent cyclic repeated",
    "连续": "continuous sustained",
    "干旱": "drought water deficit",
    "复水": "rehydration recovery rewatering",
    "浓度": "concentration",
    "测量": "measurement measured",
    "时间": "time duration",
    "处理": "treatment",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]*|\d+(?:\.\d+)?", text)]


def expand_query_terms(query: str) -> list[str]:
    additions = [english for chinese, english in DOMAIN_EXPANSIONS.items() if chinese in query]
    return tokenize(f"{query} {' '.join(additions)}")


class BM25Index:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(document)) for document in documents]
        self.lengths = [sum(counts.values()) for counts in self.term_frequencies]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency: Counter[str] = Counter()
        for counts in self.term_frequencies:
            document_frequency.update(counts.keys())
        count = len(documents)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def rank(self, terms: list[str], limit: int) -> list[tuple[int, float]]:
        query_frequency = Counter(terms)
        scores: list[tuple[int, float]] = []
        for index, counts in enumerate(self.term_frequencies):
            length_norm = 1 - self.b + self.b * self.lengths[index] / max(self.average_length, 1)
            score = 0.0
            for term, query_count in query_frequency.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                score += (
                    self.idf.get(term, 0.0)
                    * frequency
                    * (self.k1 + 1)
                    / (frequency + self.k1 * length_norm)
                    * min(query_count, 2)
                )
            if score > 0:
                scores.append((index, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:limit]


def reciprocal_rank_fusion(
    vector_ids: list[str], lexical_ids: list[str], *, rrf_k: int = 60, lexical_weight: float = 1.0
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(vector_ids, 1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + rank)
    for rank, chunk_id in enumerate(lexical_ids, 1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + lexical_weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

