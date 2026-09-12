# Development

## Project layout

See [Architecture](architecture.md) for the layer tree and the ports table.
`tests/` and `scripts/` live outside `src/` on purpose (src-layout): only
`src/` is shipped.

## Running the checks

```bash
uv run pytest                                 # full suite, in-memory doubles
uv run pytest -m retrieval                    # subset by type marker
uv run ruff check src tests scripts           # lint (E,F,Q,D,N + docstrings)
uv run ruff format --check src tests scripts  # 79 cols, single quotes
uv run mypy src tests                         # strict; bodies of untyped tests checked
uv run mutmut run                             # mutation testing on pure-logic modules
```

### Test markers

`--strict-markers` is on. File-level markers:

| Marker | Files |
|---|---|
| `indexing` | indexer, document manager, model version, multimodal |
| `retrieval` | retriever, hybrid, qdrant store, semantic cache, rewriting |
| `llm` | provider clients, router, truncation handling |
| `consumers` | RabbitMQ consumers and worker selection |
| `parsing` | markdown/pdf/docx/unstructured parsers, OCR config |
| `security` | JWT signer/validator, PII redaction |
| `evaluation` | eval runner/metrics/judge |
| `observability` | metrics, tracing, health |
| `infra` | caches, circuit breaker, locks, postgres repo, migrations |

Parser tests that need `unstructured`/libmagic skip automatically when the
native stack is unavailable (probe runs in a subprocess).

## Adding a feature end-to-end

Example: a new external dependency with graceful degradation.

1. **Port** — add a Protocol to `application/interfaces.py`:

   ```python
   @runtime_checkable
   class GeoLocator(Protocol):
       async def locate(self, address: str) -> tuple[float, float]: ...
   ```

2. **Adapter** — implement it in `infrastructure/` (lazy-import heavy deps;
   handle provider errors with a fallback).
3. **Service** — accept the port as an optional constructor parameter and
   degrade gracefully when `None`.
4. **DI** — register the provider in `container.py`, inject into the service.
5. **Config** — add pydantic fields to `config.py`; defaults must preserve
   current behavior.
6. **Tests** — a fake double next to the real adapter's tests; wire the fake
   into the affected fixtures.

Rules of thumb: `domain`/`application` never import `infrastructure`; no
feature flag default changes existing behavior; every new setting is
documented in `docs/configuration.md`.

## Type and lint policy

- `mypy --strict` over `src` and `tests`. Tests check function bodies; only
  fixture/helper *annotation strictness* is relaxed (see
  `[tool.mypy.overrides]` in `pyproject.toml`).
- Ruff: pydocstyle on `src` (module/class/method docstrings required),
  single quotes, 79 columns, isort-compatible import order.
- CI parity: `ruff check` → `ruff format --check` → `mypy src tests` →
  `pytest`.
