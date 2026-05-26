import uuid

import pytest
import tiktoken
from docmind_core.chunking import chunk_by_paragraph, chunk_by_tokens

_ENCODER = tiktoken.get_encoding("cl100k_base")


def test_empty_str() -> None:
    assert len(chunk_by_tokens("", uuid.uuid4())) == 0
    assert len(chunk_by_paragraph("", uuid.uuid4())) == 0


def test_less_than_one_chunk() -> None:
    assert len(chunk_by_tokens("hello", uuid.uuid4())) == 1
    assert len(chunk_by_paragraph("hello", uuid.uuid4())) == 1


def test_exactly_one_chunk() -> None:
    test_string = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    encoded_string = _ENCODER.encode(test_string)

    assert len(chunk_by_tokens(test_string, uuid.uuid4(), len(encoded_string), 0)) == 1
    assert len(chunk_by_paragraph(test_string, uuid.uuid4(), len(encoded_string))) == 1


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        chunk_by_tokens("hello", uuid.uuid4(), chunk_size=5, overlap=5)


def test_overlap_for_chunk_by_tokens() -> None:
    test_string = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    encoded_string = _ENCODER.encode(test_string)

    chunks = chunk_by_tokens(test_string, uuid.uuid4(), len(encoded_string) - 1, 1)

    assert len(chunks) == 2
    assert _ENCODER.encode(chunks[0].text)[-1:] == _ENCODER.encode(chunks[1].text)[:1]


def test_above_max_tokens_for_chunk_by_paragraph() -> None:
    test_string = """Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut
        aliquip ex ea commodo consequat. Lorem ipsum dolor sit amet, consectetur
        adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna
        aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi
        ut aliquip ex ea commodo consequat.

        Lorem ipsum dolor sit amet"""

    doc_id = uuid.uuid4()
    token_chunks = chunk_by_tokens(test_string, doc_id, 55)
    paragraph_chunks = chunk_by_paragraph(test_string, doc_id, 55)
    assert token_chunks[0].text == paragraph_chunks[0].text


def test_mixed_paragraphs_for_chunk_by_paragraph() -> None:
    test_string = """Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Sed do eiusmod tempor incididunt.

        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut
        aliquip ex ea commodo consequat. Lorem ipsum dolor sit amet, consectetur
        adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna
        aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi
        ut aliquip ex ea commodo consequat.

        Lorem ipsum dolor sit amet"""

    paragraph_chunks = chunk_by_paragraph(test_string, uuid.uuid4(), 60)
    assert len(paragraph_chunks) > 3
    for index, chunk in enumerate(paragraph_chunks):
        assert chunk.chunk_index == index
