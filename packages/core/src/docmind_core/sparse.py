import pickle
import re
from collections import Counter
from pathlib import Path

from rank_bm25 import BM25Okapi

from docmind_core.models import Chunk


def build_sparse_index(chunks: list[Chunk]) -> tuple[BM25Okapi, dict[str, int]]:
    vocab: dict[str, int] = {}
    tokenized_docs = []
    for chunk in chunks:
        tokens = re.findall(r"\b\w+\b", chunk.text.lower())
        tokenized_docs.append(tokens)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    return (BM25Okapi(tokenized_docs), vocab)


def chunk_to_sparse_vector(
    text: str, bm25: BM25Okapi, vocab: dict[str, int]
) -> tuple[list[int], list[float]]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    term_freqs = Counter(tokens)
    doc_len = len(tokens)

    indices: list[int] = []
    weights: list[float] = []

    for term, tf in term_freqs.items():
        idf = bm25.idf.get(term)
        if idf is None or term not in vocab:
            continue

        numerator = tf * (bm25.k1 + 1)
        denominator = tf + bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)

        indices.append(vocab[term])
        weights.append(idf * numerator / denominator)

    return (indices, weights)


def save_sparse_index(bm25: BM25Okapi, vocab: dict[str, int], path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "vocab": vocab}, f)


def load_sparse_index(path: Path) -> tuple[BM25Okapi, dict[str, int]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["vocab"]
