# Agentic Retrieval (Corrective RAG)

Agentic retrieval closes the loop between *search* and *answer quality*:
an optional **planner** decomposes complex questions before the search,
an optional **grader** judges the retrieved hits, and a negative verdict
triggers a corrective query rewrite and another search round.

Everything is **opt-in**; with the defaults (`AGENTIC_GRADER_ENABLED=false`,
`AGENTIC_PLANNER_ENABLED=false`) retrieval behaves exactly as described in
[Retrieval](retrieval.md).

## How the loop works

```
question
  |
  v
[planner]  question -> [sub-question 1, sub-question 2, ...]   (optional)
  |            parallel searches, fused with RRF
  v
[search round] -> hits -> score threshold -> parent auto-merge
  |
  v
[grader]   relevant? -> answer                                  (optional)
  | no, and rounds left
  v
rewrite(question, feedback=verdict.reason) -> next round
  | rounds exhausted
  v
answer from the best hits available
```

- **Round 0 without a grader** is the classic single-pass pipeline.
- The **original question** (never rewrites or sub-queries) is persisted in
  the conversation history.
- The loop never exceeds `AGENTIC_MAX_ROUNDS` search rounds; when exhausted,
  the service still answers from the best hits rather than failing.

## Ports and adapters

| Port (`application/interfaces.py`) | Adapters (`infrastructure/agentic.py`) |
|---|---|
| `QueryPlanner.plan(query) -> list[str]` | `LLMQueryPlanner`, `NoOpQueryPlanner` |
| `DocumentGrader.grade(query, hits) -> GradeVerdict` | `LLMDocumentGrader`, `NoOpDocumentGrader` |

`GradeVerdict` is a frozen dataclass: `relevant: bool`, `reason: str`,
`score: float`. The `reason` of a negative verdict is passed to the rewriter
as corrective feedback (see `QueryRewriter.rewrite(query, feedback=...)`).

**Graceful degradation is part of every contract**: the LLM planner falls
back to the original question, the LLM grader falls back to a score-threshold
gate (`fallback_threshold`, wired to `SEARCH_SCORE_THRESHOLD` semantics), and
neither adapter ever raises into the retrieval loop.

## Sub-question fusion

When a planner returns multiple sub-questions, they are embedded and searched
**in parallel** (`asyncio.gather`) and fused with Reciprocal Rank Fusion
(the same `k` as `SEARCH_HYBRID_RRF_K`). The payload of the first occurrence
of a chunk wins; fused scores replace the raw similarity in the hit list.
`AGENTIC_SUBQUERY_LIMIT` caps how many sub-queries are used per round.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AGENTIC_GRADER_ENABLED` | `false` | Grade hits each round; rewrite on negative verdicts |
| `AGENTIC_PLANNER_ENABLED` | `false` | Split the question into sub-queries |
| `AGENTIC_MAX_ROUNDS` | `0` | Max search rounds when a grader is wired (`0`/no grader = single pass) |
| `AGENTIC_SUBQUERY_LIMIT` | `3` | Max sub-queries per round |

## Cost model

Each enabled feature costs one auxiliary LLM call per round:
planning (once per query), grading (once per round). With both enabled and
`AGENTIC_MAX_ROUNDS=2`, a query can spend up to **5** auxiliary LLM calls
(1 plan + 2 grades + up to 2 rewrites) plus the final generation. Cache hits
from the [semantic cache](retrieval.md#semantic-answer-cache) skip the whole
loop.

## Testing

`tests/test_agentic_retrieval.py` (marker `retrieval`) covers the round
machine (bad round -> rewrite -> good round, rounds exhaustion), sub-query
fusion and limits, JSON verdict/plan parsing and threshold fallbacks.
