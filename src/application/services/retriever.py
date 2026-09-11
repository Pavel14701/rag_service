"""RAG retrieval and answer generation service."""

import structlog
from typing import Any, Callable

from application.interfaces import (
    VectorStore,
    DocumentRepository,
    EmbeddingModel,
    LLMGenerator,
)

logger = structlog.get_logger(__name__)


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
        hybrid_enabled: bool = False,
        hybrid_rrf_k: int = 60,
        pii_redactor: Callable[[str], str] | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._repo = repo
        self._embedding = embedding
        self._llm = llm
        self._default_top_k = default_top_k
        self._default_temperature = default_temperature
        self._hybrid_enabled = hybrid_enabled
        self._hybrid_rrf_k = hybrid_rrf_k
        # Optional callable applied to query/answer before persisting
        # conversations (PII redaction, see shared.pii).
        self._pii_redactor = pii_redactor

    async def answer_query(
        self,
        user_id: str,
        query: str,
        conversation_id: str | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        user_groups: list[str] | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> dict[str, Any]:
        """Process a user query: retrieve context and answer it.

        Args:
            user_id: ID of the requesting user.
            query: User's question.
            conversation_id: Optional conversation ID for context.
            top_k: Override default number of chunks to retrieve.
            temperature: Override default LLM temperature.
            user_groups: Access groups of the user (e.g. from the JWT
                ``groups`` claim). When ``None``, groups are loaded from
                the repository (``user_groups`` table).
            llm_provider: LLM provider for this request (per-request
                routing; must be among ``LLM_ENABLED_PROVIDERS``).
            llm_model: LLM model override for this request (must be in
                ``LLM_ALLOWED_MODELS`` when the allowlist is set).

        Returns:
            Dictionary with 'answer', 'sources', and 'conversation_id'.

        """
        if not query.strip():
            return {
                'answer': 'Please provide a non-empty question.',
                'sources': [],
                'conversation_id': conversation_id,
            }

        top_k = self._default_top_k if top_k is None else top_k
        temperature = (
            self._default_temperature if temperature is None else temperature
        )

        # Access control: owner_id == user_id OR access_group in user's groups.
        # Explicit user_groups (e.g. from the JWT claim) take precedence over
        # the repository-backed membership table.
        if user_groups is not None:
            groups = user_groups
        else:
            groups = await self._repo.get_user_groups(user_id)
        filter_cond: dict[str, Any]
        if groups:
            filter_cond = {
                'should': [
                    {'key': 'owner_id', 'match': {'value': user_id}},
                    {'key': 'access_group', 'match': {'value': groups}},
                ]
            }
        else:
            filter_cond = {'key': 'owner_id', 'match': {'value': user_id}}

        # Embed query (E5 models require the "query: " prefix)
        query_vec = (await self._embedding.embed_query([query]))[0]

        # Search (hybrid: vector ranking fused with lexical BM25 when enabled)
        hits = await self._vector_store.search(
            vector=query_vec,
            top_k=top_k,
            filter_condition=filter_cond,
            keyword_query=query if self._hybrid_enabled else None,
        )

        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        for hit in hits:
            text = hit.get('text', '')
            if text:
                context_parts.append(text)
            payload = hit.get('payload', {})
            sources.append(
                {
                    'doc_id': payload.get('doc_id'),
                    'chunk_id': hit.get('id'),
                    'page': payload.get('page'),
                }
            )

        if not context_parts:
            logger.info('no_relevant_context', user_id=user_id)
            answer = "I don't have enough information to answer that."
            context = ''
        else:
            context = '\n\n'.join(context_parts)
            system_prompt = self._build_system_prompt()
            user_prompt = (
                f'Context:\n{context}\n\nQuestion: {query}\n\nAnswer:'
            )
            answer = await self._generate(
                system_prompt,
                user_prompt,
                temperature,
                llm_provider,
                llm_model,
            )

        # Persist the conversation with PII redacted when configured:
        # the raw user question and the generated answer may contain
        # personal data, sources are doc metadata only.
        stored_query = (
            self._pii_redactor(query) if self._pii_redactor else query
        )
        stored_answer = (
            self._pii_redactor(answer) if self._pii_redactor else answer
        )
        await self._repo.save_conversation(
            user_id, stored_query, stored_answer, sources
        )

        return {
            'answer': answer,
            'sources': sources,
            'conversation_id': conversation_id,
        }

    async def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        llm_provider: str | None,
        llm_model: str | None,
    ) -> str:
        """Generate via the requested provider/model when a routing request
        is made; falls back to the default single-provider path otherwise.
        """
        if llm_provider or llm_model:
            generate_with = getattr(self._llm, 'generate_with', None)
            if generate_with is None:
                raise ValueError(
                    'Per-request LLM routing requires a multi-provider router '
                    '(set LLM_ENABLED_PROVIDERS).'
                )
            answer: str = await generate_with(
                provider=llm_provider,
                model=llm_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
            return answer
        return await self._llm.generate(
            system_prompt, user_prompt, temperature
        )

    def _build_system_prompt(self) -> str:
        return (
            'You are a helpful assistant that answers questions '
            'based strictly on the provided context.\n'
            'Rules:\n'
            '1. Use ONLY information from the context to answer.\n'
            "2. If the answer is not in the context, say 'I don't know'.\n"
            '3. Do not use any external knowledge or prior training data.\n'
            '4. If the context contains contradictory information, '
            'mention that.\n'
            '5. If appropriate, cite the source '
            "(e.g., 'according to document X')."
        )
