"""Prometheus metrics for the rag_service."""

from prometheus_client import Counter, Histogram

MESSAGE_PROCESSING_SECONDS = Histogram(
    'rag_message_processing_seconds',
    'Time spent processing a message, by queue.',
    ['queue'],
    buckets=(0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 60.0, 300.0),
)
MESSAGES_TOTAL = Counter(
    'rag_messages_total',
    'Processed messages by queue and outcome (success/error/retry/dlq).',
    ['queue', 'status'],
)
VECTOR_SEARCH_SECONDS = Histogram(
    'rag_vector_search_seconds',
    'Latency of vector store search calls.',
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)
LLM_GENERATION_SECONDS = Histogram(
    'rag_llm_generation_seconds',
    'Latency of LLM generation calls.',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)
LLM_TOKENS_TOTAL = Counter(
    'rag_llm_tokens_total',
    'LLM tokens consumed, by kind (prompt/completion).',
    ['kind'],
)
LLM_CACHE_TOTAL = Counter(
    'rag_llm_cache_total',
    'LLM answer cache lookups, by result (hit/miss).',
    ['result'],
)
