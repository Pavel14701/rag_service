# Retrieval Pipeline

End-to-end path of one question, and every quality lever along it.

## 1. Optional query rewriting

`QUERY_REWRITE_ENABLED=true` adds one LLM call that normalizes the question
(expands abbreviations, fixes typos). `LLMQueryRewriter` degrades gracefully —
any failure returns the original query. The **original** question is what gets
persisted in the conversation history.

## 2. ACL-filtered search

The embedding is computed for the (possibly rewritten) query; the search runs
with a Qdrant filter:

```
should: [ owner_id == user, access_group in user_groups ]
```

Unauthorized chunks never reach the prompt. Groups come from the JWT `groups`
claim or the `user_groups` table.

## 3. Hybrid search

`SEARCH_HYBRID=true` adds a lexical leg: full-text candidates from the Qdrant
`text` index are scored with BM25 and fused with the vector ranking via RRF
(`SEARCH_HYBRID_RRF_K`). Noise guards:

- `SEARCH_HYBRID_STOPWORDS` — stop words stripped from the keyword query;
- `SEARCH_BM25_SCORE_THRESHOLD` — weak lexical candidates never enter fusion.

If the lexical leg yields nothing after the guards, the search degrades to
pure vector — availability first.

## 4. Context packing guards

- `SEARCH_SCORE_THRESHOLD` drops weak vector hits before they spend budget
  (use `SEARCH_BM25_SCORE_THRESHOLD` for the hybrid leg — RRF scores have a
  different scale).
- `SEARCH_CONTEXT_MAX_CHARS` stops packing once the budget is spent, so the
  prompt leaves room for the answer inside `LLM_MAX_TOKENS`.

## 5. Parent-Child auto-merge

`PARENT_CHILD_CHILD_CHARS > 0` changes indexing: each element is split into
parents (`max_tokens`) and children (`child_chars`); children are embedded and
store `parent_id` + `parent_text` in their payload. At query time sibling hits
of the same parent collapse into one hit carrying the **parent text** — small
chunks are precise for embedding, the LLM still sees coherent context.

## 6. Prompt-injection isolation

The packed context is wrapped in `<context>...</context>` and the system
prompt states that everything inside is untrusted data: instructions found in
retrieved documents are never followed.

## 7. Query-answer caching

Two independent layers:

- **Exact** (`REDIS_URL`): answers keyed by
  `(system_prompt, user_prompt, temperature)`; embeddings keyed by text.
- **Semantic** (`SEMANTIC_CACHE_ENABLED`): cosine similarity over stored query
  vectors; a hit saves the LLM call entirely. **Cache-then-validate**: each
  entry stores provenance (source chunk ids, embedding model, LLM provider/
  model) and is trusted only when the meta matches the current request AND
  every source chunk is still visible under the requester ACL filter
  (`retrieve_by_ids` with the same filter). `SEMANTIC_CACHE_STRICT_ACL=true\
  additionally requires an exact access-group set match (hashed `acl_key`).

> Multi-tenant note: with cache-then-validate the ACL leak is closed at
> chunk granularity; strict mode covers answer-level group sensitivity.

## 8. Truncation handling

`finish_reason=length` / `stop_reason=max_tokens`:

- the partial answer is **never cached**;
- with `LLM_CONTINUE_ON_TRUNCATION=true` up to two continuation requests are
  issued and the parts are concatenated into the full answer.

## 9. Indexing-side quality levers

| Lever | Setting | Effect |
|---|---|---|
| Tiny-chunk merge | `CHUNK_MIN_CHARS` | no stub chunks in the index |
| Table chunking | automatic | row-groups + repeated header row (pipe text + HTML) |
| Parent-Child | `PARENT_CHILD_CHILD_CHARS` | precise matching, coherent context |
| Parse budget | `PARSE_TIMEOUT` | hung OCR becomes a permanent per-file failure |
| Parse isolation | `PARSE_ISOLATION_ENABLED` | OCR runs in a killable subprocess: on timeout the whole process tree is killed instead of leaking a zombie thread |
| MIME validation | `PARSE_VALIDATE_MIME` | binary-garbage files rejected before OCR |

## Tuning recipe

1. Index the corpus, run `scripts/run_eval.py` for the baseline.
2. Enable hybrid search + stop words, re-run, compare with `compare_reports`.
3. Add Parent-Child and query rewriting; watch latency vs faithfulness.
4. Only then enable the semantic cache (after checking the ACL caveat).
