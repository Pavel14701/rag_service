"""RAG retrieval and answer generation service."""

import logging
from typing import Any

from application.interfaces import (
    VectorStore,
    DocumentRepository,
    EmbeddingModel,
    LLMGenerator,
)
from domain.exceptions import PermissionDeniedError

logger = logging.getLogger(__name__)


class RetrieverService:
    """Service to answer user queries using RAG."""

    def __init__(
        self,
        vector_store: VectorStore,
        repo: DocumentRepository,
        embedding: EmbeddingModel,
        llm: LLMGenerator,
        default_top_k: int = 5,
        default_temperature: float = 0.1,
    ) -> None:
        self._vector_store = vector_store
        self._repo = repo
        self._embedding = embedding
        self._llm = llm
        self._default_top_k = default_top_k
        self._default_temperature = default_temperature

    async def answer_query(
        self,
        user_id: str,
        query: str,
        conversation_id: str | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Process a user query: retrieve relevant context and generate an answer.

        Args:
            user_id: ID of the requesting user.
            query: User's question.
            conversation_id: Optional conversation ID for context.
            top_k: Override default number of chunks to retrieve.
            temperature: Override default LLM temperature.

        Returns:
            Dictionary with 'answer', 'sources', and 'conversation_id'.
        """
        if not query.strip():
            return {
                "answer": "Please provide a non-empty question.",
                "sources": [],
                "conversation_id": conversation_id,
            }

        top_k = self._default_top_k if top_k is None else top_k
        temperature = self._default_temperature if temperature is None else temperature

        # Access control: owner_id == user_id OR access_group in user's groups
        groups = await self._repo.get_user_groups(user_id)
        if groups:
            filter_cond = {
                "should": [
                    {"key": "owner_id", "match": {"value": user_id}},
                    {"key": "access_group", "match": {"value": groups}},
                ]
            }
        else:
            filter_cond = {"key": "owner_id", "match": {"value": user_id}}

        # Embed query
        query_vec = (await self._embedding.embed([query]))[0]

        # Search
        hits = await self._vector_store.search(
            vector=query_vec,
            top_k=top_k,
            filter_condition=filter_cond,
        )

        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        for hit in hits:
            text = hit.get("text", "")
            if text:
                context_parts.append(text)
            payload = hit.get("payload", {})
            sources.append({
                "doc_id": payload.get("doc_id"),
                "chunk_id": hit.get("id"),
                "page": payload.get("page"),
            })

        if not context_parts:
            logger.info(f"No relevant context for user {user_id}")
            answer = "I don't have enough information to answer that."
            context = ""
        else:
            context = "\n\n".join(context_parts)
            system_prompt = self._build_system_prompt()
            user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            answer = await self._llm.generate(system_prompt, user_prompt, temperature)

        await self._repo.save_conversation(user_id, query, answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "conversation_id": conversation_id,
        }

    def _build_system_prompt(self) -> str:
        return (
            "You are a helpful assistant that answers questions based strictly on the provided context.\n"
            "Rules:\n"
            "1. Use ONLY information from the context to answer.\n"
            "2. If the answer is not in the context, say 'I don't know'.\n"
            "3. Do not use any external knowledge or prior training data.\n"
            "4. If the context contains contradictory information, mention that.\n"
            "5. If appropriate, cite the source (e.g., 'according to document X')."
        )
