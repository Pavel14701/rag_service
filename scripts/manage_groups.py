"""CLI for managing user access groups (ACL membership table).

Usage (from the project root)::

    uv run python scripts/manage_groups.py set <user_id> group1,group2
    uv run python scripts/manage_groups.py list <user_id>

Groups are used by the vector-store ACL filter (``access_group`` claim)
when the JWT does not carry a ``groups`` claim.
"""

import asyncio
import sys

sys.path.insert(0, "src")

from config import Settings  # noqa: E402
from infrastructure.repositories.postgres_repo import (  # noqa: E402
    PostgresDocumentRepository,
)
from sqlalchemy.ext.asyncio import (  # noqa: E402
    async_sessionmaker,
    create_async_engine,
    AsyncSession,
)


def _repo(settings: Settings) -> PostgresDocumentRepository:
    engine = create_async_engine(settings.postgres_dsn, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return PostgresDocumentRepository(session_factory)


async def _set(user_id: str, groups_csv: str) -> None:
    groups = [g.strip() for g in groups_csv.split(",") if g.strip()]
    repo = _repo(Settings())
    await repo.set_user_groups(user_id, groups)
    print(f"user {user_id!r}: groups set to {groups}")


async def _list(user_id: str) -> None:
    repo = _repo(Settings())
    groups = await repo.get_user_groups(user_id)
    print(f"user {user_id!r}: {groups if groups else '(no groups)'}")


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in {"set", "list"}:
        print(__doc__)
        sys.exit(2)
    command, user_id = sys.argv[1], sys.argv[2]
    if command == "set" and len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    if command == "set":
        asyncio.run(_set(user_id, sys.argv[3]))
    else:
        asyncio.run(_list(user_id))


if __name__ == "__main__":
    main()