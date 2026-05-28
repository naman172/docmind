import uuid

import tiktoken

from docmind_core.models import Chunk

_ENCODER = tiktoken.get_encoding("cl100k_base")


def chunk_by_tokens(
    text: str,
    document_id: uuid.UUID,
    chunk_size: int = 512,
    overlap: int = 50,
    chunk_index_start: int = 0,
) -> list[Chunk]:
    encoded_string = _ENCODER.encode(text)
    chunks: list[Chunk] = []

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    for i in range(0, len(encoded_string), chunk_size - overlap):
        string_partition = encoded_string[i : i + chunk_size]
        decoded_string = _ENCODER.decode(string_partition)
        chunks.append(
            Chunk(
                document_id=document_id,
                text=decoded_string,
                chunk_index=chunk_index_start + len(chunks),
                token_count=len(string_partition),
            )
        )

    return chunks


def chunk_by_paragraph(
    text: str,
    document_id: uuid.UUID,
    max_tokens: int = 512,
) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []

    for p in paragraphs:
        encoded_string = _ENCODER.encode(p)
        if len(encoded_string) <= max_tokens:
            chunks.append(
                Chunk(
                    document_id=document_id,
                    text=p,
                    chunk_index=len(chunks),
                    token_count=len(encoded_string),
                )
            )
        else:
            chunks_for_p = chunk_by_tokens(
                p, document_id, max_tokens, chunk_index_start=len(chunks)
            )
            for chunk in chunks_for_p:
                chunks.append(chunk)
    return chunks


def chunk_fixed(
    text: str,
    document_id: uuid.UUID,
    source_file: str | None,
    chunk_size: int = 512,
    overlap: int = 50,
    min_new_chars: int = 25,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    for i in range(0, len(text), chunk_size - overlap):
        if i + overlap >= len(text) - min_new_chars:
            break

        chunks.append(
            Chunk(
                document_id=document_id,
                source_file=source_file,
                text=text[i : i + chunk_size],
                chunk_index=len(chunks),
            )
        )

    return chunks


def _split_text(
    text: str, separators: list[str], chunk_size: int, overlap: int
) -> list[str]:
    if not separators:
        return [
            c.text for c in chunk_fixed(text, uuid.uuid4(), None, chunk_size, overlap)
        ]

    current_separator = separators[0]
    remaining_separators = separators[1:]

    pieces = [p.strip() for p in text.split(current_separator) if p.strip()]

    result = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            result.append(piece)
        else:
            result.extend(_split_text(piece, remaining_separators, chunk_size, overlap))

    return result


def chunk_recursive(
    text: str,
    document_id: uuid.UUID,
    source_file: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[Chunk]:
    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " "]

    pieces = _split_text(text, SEPARATORS, chunk_size, overlap)

    return [
        Chunk(
            document_id=document_id,
            source_file=source_file,
            text=piece,
            chunk_index=i,
        )
        for i, piece in enumerate(pieces)
    ]


def chunk_semantic(
    text: str,
    document_id: uuid.UUID,
    source_file: str,
    chunk_size: int = 512,
    similarity_threshold: float = 0.5,
) -> list[Chunk]:
    """
    Semantic chunking using embedding similarity between adjacent sentences.

    Algorithm (not yet implemented):
    1. Split text into sentences
    2. Embed each sentence using the configured embedding model
    3. Compute cosine similarity between adjacent sentence embeddings
    4. Split where similarity drops below similarity_threshold
    5. Merge small resulting chunks up to chunk_size

    Deferred to when embedding client is available in benchmark context.
    Cost: O(n) embedding calls where n = number of sentences. Expensive on
    large corpora.
    """
    raise NotImplementedError("chunk_semantic requires embedding client")
