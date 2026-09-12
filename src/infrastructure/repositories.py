"""PostgreSQL implementation of DocumentRepository (SQLAlchemy asyncio)."""

import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select, update, and_, Index, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB

from domain.model import Document, DocStatus
from application.interfaces import DocumentRepository


# SQLAlchemy ORM Models
class Base(DeclarativeBase):
    """Declarative SQLAlchemy base."""

    pass


class DocumentORM(Base):
    """ORM table for document metadata."""

    __tablename__ = 'documents'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    file_name: Mapped[str] = mapped_column(nullable=False)
    file_path: Mapped[str] = mapped_column(nullable=False)
    file_hash: Mapped[str] = mapped_column(nullable=False)
    owner_id: Mapped[str] = mapped_column(nullable=False, index=True)
    access_group: Mapped[str | None] = mapped_column(nullable=True, index=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    status: Mapped[str] = mapped_column(nullable=False)
    deleted: Mapped[bool] = mapped_column(default=False)

    # Partial index: owner/ACL lookups never need soft-deleted rows.
    __table_args__ = (
        Index(
            'ix_documents_owner_active',
            'owner_id',
            postgresql_where=text('deleted = false'),
        ),
    )


class ConversationORM(Base):
    """ORM table for stored conversations."""

    __tablename__ = 'conversations'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(nullable=False, index=True)
    query: Mapped[str] = mapped_column(nullable=False)
    response: Mapped[str] = mapped_column(nullable=False)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )


class UserGroupORM(Base):
    """User membership in access groups (managed via ``set_user_groups``)."""

    __tablename__ = 'user_groups'

    user_id: Mapped[str] = mapped_column(primary_key=True)
    group_name: Mapped[str] = mapped_column(primary_key=True)


class PostgresDocumentRepository(DocumentRepository):
    """PostgreSQL adapter using SQLAlchemy asyncio.

    An optional ``read_session_factory`` (pointing at a read replica)
    is used for read queries; writes always go to the primary.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        read_session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._read_session_factory = read_session_factory or session_factory

    async def get_document(self, doc_id: uuid.UUID) -> Document | None:
        """Retrieve a document by its ID."""
        async with self._read_session_factory() as session:
            stmt = select(DocumentORM).where(
                and_(DocumentORM.id == doc_id, DocumentORM.deleted.is_(False))
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return None if row is None else self._to_domain(row)

    async def get_all_active(self) -> list[Document]:
        """Get all non-deleted documents."""
        async with self._read_session_factory() as session:
            stmt = select(DocumentORM).where(DocumentORM.deleted.is_(False))
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._to_domain(row) for row in rows]

    async def iter_active_documents(
        self, batch_size: int = 500
    ) -> AsyncIterator[Document]:
        """Keyset pagination over non-deleted documents."""
        async with self._read_session_factory() as session:
            last_id: uuid.UUID | None = None
            while True:
                if last_id is None:
                    stmt = (
                        select(DocumentORM)
                        .where(DocumentORM.deleted.is_(False))
                        .order_by(DocumentORM.id)
                        .limit(batch_size)
                    )
                else:
                    stmt = (
                        select(DocumentORM)
                        .where(
                            and_(
                                DocumentORM.deleted.is_(False),
                                DocumentORM.id > last_id,
                            )
                        )
                        .order_by(DocumentORM.id)
                        .limit(batch_size)
                    )
                rows = (await session.execute(stmt)).scalars().all()
                if not rows:
                    return
                for row in rows:
                    yield self._to_domain(row)
                last_id = rows[-1].id

    async def save(self, document: Document) -> None:
        """Persist or update a document record."""
        async with self._session_factory() as session:
            orm = self._from_domain(document)
            session.add(orm)
            await session.commit()

    async def update_status(
        self, doc_id: uuid.UUID, status: DocStatus
    ) -> None:
        """Update the indexing status of a document."""
        async with self._session_factory() as session:
            stmt = (
                update(DocumentORM)
                .where(DocumentORM.id == doc_id)
                .values(status=status.value)
            )
            await session.execute(stmt)
            await session.commit()

    async def mark_deleted(self, doc_id: uuid.UUID) -> None:
        """Soft-delete a document."""
        async with self._session_factory() as session:
            stmt = (
                update(DocumentORM)
                .where(DocumentORM.id == doc_id)
                .values(deleted=True)
            )
            await session.execute(stmt)
            await session.commit()

    async def get_user_groups(self, user_id: str) -> list[str]:
        """Return the ACL groups the user belongs to."""
        async with self._read_session_factory() as session:
            stmt = select(UserGroupORM.group_name).where(
                UserGroupORM.user_id == user_id
            )
            rows = (await session.execute(stmt)).scalars().all()
            return list(rows)

    async def set_user_groups(self, user_id: str, groups: list[str]) -> None:
        """Replace the ACL group membership of the user."""
        async with self._session_factory() as session:
            await session.execute(
                delete(UserGroupORM).where(UserGroupORM.user_id == user_id)
            )
            for group in groups:
                session.add(UserGroupORM(user_id=user_id, group_name=group))
            await session.commit()

    async def save_conversation(
        self,
        user_id: str,
        query: str,
        response: str,
        sources: list[dict[str, Any]],
    ) -> None:
        """Store a conversation record."""
        async with self._session_factory() as session:
            conv = ConversationORM(
                user_id=user_id,
                query=query,
                response=response,
                sources=sources,
            )
            session.add(conv)
            await session.commit()

    @staticmethod
    def _to_domain(orm: DocumentORM) -> Document:
        return Document(
            id=orm.id,
            file_name=orm.file_name,
            file_path=orm.file_path,
            file_hash=orm.file_hash,
            owner_id=orm.owner_id,
            access_group=orm.access_group,
            uploaded_at=orm.uploaded_at,
            status=DocStatus(orm.status),
            deleted=orm.deleted,
        )

    @staticmethod
    def _from_domain(domain: Document) -> DocumentORM:
        return DocumentORM(
            id=domain.id,
            file_name=domain.file_name,
            file_path=domain.file_path,
            file_hash=domain.file_hash,
            owner_id=domain.owner_id,
            access_group=domain.access_group,
            uploaded_at=domain.uploaded_at,
            status=domain.status.value,
            deleted=domain.deleted,
        )
