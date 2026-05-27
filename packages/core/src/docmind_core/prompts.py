RAG_SYSTEM_PROMPT_V1: str = """
You are a helpful AI assistant.

Answer the user's query using ONLY the provided context.
If the answer is not contained in the context, say you don't know.

Context:
{context}
"""


def build_rag_prompt(context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return str(RAG_SYSTEM_PROMPT_V1.format(context=context))
