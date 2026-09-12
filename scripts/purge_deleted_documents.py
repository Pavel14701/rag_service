"""Hard-delete soft-deleted documents older than N days (ops cron job).

Soft deletes keep rows forever (recovery safety); this script reclaims
the storage once the retention window passes.

Usage (from the project root)::

    uv run python scripts/purge_deleted_documents.py --days 30 [--apply]

Without ``--apply`` the script only reports how many rows it would purge.
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, 'src')

from config import Settings  # noqa: E402
from infrastructure.repositories import (  # noqa: E402
    DocumentORM,
)
from sqlalchemy import delete, select, func  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
    AsyncSession,
)


def _session_factory(settings: Settings) -> async_sessionmaker[AsyncSession]:
    engine: AsyncEngine = create_async_engine(
        settings.postgres_dsn, echo=False
    )
    return async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )


async def purge(days: int, apply: bool) -> None:
    """Count (or delete) soft-deleted documents older than ``days``."""
    settings = Settings()
    factory = _session_factory(settings)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    async with factory() as session:
        stmt = (
            select(func.count())
            .select_from(DocumentORM)
            .where(
                DocumentORM.deleted.is_(True),
                DocumentORM.uploaded_at < cutoff,
            )
        )
        count = (await session.execute(stmt)).scalar_one()
        if not apply:
            print(f'{count} soft-deleted documents older than {days} days')
            print('re-run with --apply to purge them')
            return
        await session.execute(
            delete(DocumentORM).where(
                DocumentORM.deleted.is_(True),
                DocumentORM.uploaded_at < cutoff,
            )
        )
        await session.commit()
        print(f'purged {count} soft-deleted documents older than {days} days')


def main() -> None:
    """CLI entry point: report or purge stale soft-deleted rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument(
        '--apply', action='store_true', help='actually delete rows'
    )
    args = parser.parse_args()
    asyncio.run(purge(args.days, args.apply))


if __name__ == '__main__':
    main()
