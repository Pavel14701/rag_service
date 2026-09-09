"""PostgreSQL implementation of DocumentRepository using SQLAlchemy asyncio."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB

from domain.entities.document import Document, DocStatus
from application.interfaces import DocumentRepository


# SQLAlchemy ORM Models
class Base(DeclarativeBase):
    pass


class DocumentORM(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name: Mapped[str] = mapped_column(nullable=False)
    file_path: Mapped[str] = mapped_column(nullable=False)
    file_hash: Mapped[str] = mapped_column(nullable=False)
    owner_id: Mapped[str] = mapped_column(nullable=False, index=True)
    access_group: Mapped[str | None] = mapped_column(nullable=True, index=True)
    uploaded_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    status: Mapped[str] = mapped_column(nullable=False)
    deleted: Mapped[bool] = mapped_column(default=False)


class ConversationORM(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(nullable=False, index=True)
    query: Mapped[str] = mapped_column(nullable=False)
    response: Mapped[str] = mapped_column(nullable=False)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class PostgresDocumentRepository(DocumentRepository):
    """PostgreSQL adapter using SQLAlchemy asyncio."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_document(self, doc_id: uuid.UUID) -> Document | None:
        async with self._session_factory() as session:
            stmt = select(DocumentORM).where(
                and_(DocumentORM.id == doc_id, not DocumentORM.deleted)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return self._to_domain(row)

    async def get_all_active(self) -> list[Document]:
        async with self._session_factory() as session:
            stmt = select(DocumentORM).where(not DocumentORM.deleted)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._to_domain(row) for row in rows]

    async def save(self, document: Document) -> None:
        async with self._session_factory() as session:
            orm = self._from_domain(document)
            session.add(orm)
            await session.commit()

    async def update_status(self, doc_id: uuid.UUID, status: DocStatus) -> None:
        async with self._session_factory() as session:
            stmt = (
                update(DocumentORM)
                .where(DocumentORM.id == doc_id)
                .values(status=status.value)
            )
            await session.execute(stmt)
            await session.commit()

    async def mark_deleted(self, doc_id: uuid.UUID) -> None:
        async with self._session_factory() as session:
            stmt = (
                update(DocumentORM)
                .where(DocumentORM.id == doc_id)
                .values(deleted=True)
            )
            await session.execute(stmt)
            await session.commit()

    async def get_user_groups(self, user_id: str) -> list[str]:
        # Placeholder
        return []

    async def save_conversation(
        self,
        user_id: str,
        query: str,
        response: str,
        sources: list[dict[str, Any]],
    ) -> None:
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