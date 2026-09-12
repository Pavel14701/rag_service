"""Index the eval corpus with deterministic document ids.

Reads every ``eval/corpus/*.md`` file, uploads it to MinIO under
``eval/<name>``, saves a Document whose id is derived deterministically
(``uuid5(NAMESPACE_URL, 'eval-corpus:<name>')``) and indexes it — so the
``relevant_doc_ids`` in ``eval/golden_dataset.json`` match real document
ids without any manual bookkeeping.

Usage (from the project root)::

    PYTHONPATH=src uv run python scripts/index_eval_corpus.py
    PYTHONPATH=src uv run python scripts/index_eval_corpus.py --force

Requires the backing services (MinIO, Qdrant, Postgres) to be running;
embedding model files are downloaded on first use.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from application.interfaces import DocumentRepository, FileStorage  # noqa: E402
from application.services import IndexerService  # noqa: E402
from config import Settings  # noqa: E402
from container import create_container  # noqa: E402
from domain.model import DocStatus, Document  # noqa: E402


NAMESPACE = uuid.NAMESPACE_URL
PREFIX = 'eval-corpus:'


def corpus_id(name: str) -> uuid.UUID:
    """Deterministic document id for a corpus file name."""
    return uuid.uuid5(NAMESPACE, PREFIX + name)


async def index_corpus(force: bool) -> int:
    """Index every corpus file; returns the number of indexed documents."""
    corpus_dir = Path(__file__).resolve().parents[1] / 'eval' / 'corpus'
    files = sorted(corpus_dir.glob('*.md'))
    if not files:
        print(f'No corpus files found in {corpus_dir}')
        return 0

    container = create_container()
    _ = Settings  # resolved transitively through the container graph
    container = create_container()
    storage = await container.get(FileStorage)
    repo = await container.get(DocumentRepository)
    indexer = await container.get(IndexerService)

    indexed = 0
    for path in files:
        doc_id = corpus_id(path.name)
        existing = await repo.get_document(doc_id)
        if existing is not None and not force:
            print(f'skip (already indexed, use --force): {path.name}')
            continue

        storage_key = f'eval/{path.name}'
        await storage.upload_file(path, storage_key)
        await repo.save(
            Document(
                id=doc_id,
                file_name=path.name,
                file_path=storage_key,
                file_hash=f'eval-{path.name}',
                owner_id='eval-user',
                access_group=None,
                uploaded_at=datetime.now(timezone.utc),
                status=DocStatus.PENDING,
            )
        )
        await indexer.index_document(doc_id)
        indexed += 1
        print(f'indexed {path.name} -> {doc_id}')

    _ = Settings  # settings are resolved transitively via the container
    return indexed


def main() -> None:
    """CLI entry point: index the corpus (or report what is indexed)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--force', action='store_true', help='re-index already indexed files'
    )
    args = parser.parse_args()
    indexed = asyncio.run(index_corpus(force=args.force))
    print(f'done: {indexed} document(s) indexed')


if __name__ == '__main__':
    main()
