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
    chunks = []

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    for i in range(0, len(encoded_string), chunk_size - overlap):
        chunks.append(encoded_string[i : i + chunk_size])

    chunks[:] = [
        Chunk(
            document_id=document_id,
            text=_ENCODER.decode(x),
            chunk_index=chunk_index_start + i,
            token_count=len(x),
        )
        for i, x in enumerate(chunks)
    ]

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
