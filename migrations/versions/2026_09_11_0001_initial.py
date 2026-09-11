"""initial schema: documents, conversations

Revision ID: 0001
Revises:
Create Date: 2026-09-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("file_hash", sa.String(), nullable=False),
        sa.Column("owner_id", sa.String(), nullable=False),
        sa.Column("access_group", sa.String(), nullable=True),
        sa.Column("uploaded_at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_documents_owner_id"), "documents", ["owner_id"], unique=False
    )
    op.create_index(
        op.f("ix_documents_access_group"), "documents", ["access_group"], unique=False
    )

    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("query", sa.String(), nullable=False),
        sa.Column("response", sa.String(), nullable=False),
        sa.Column("sources", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_conversations_user_id"), "conversations", ["user_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_conversations_user_id"), table_name="conversations")
    op.drop_table("conversations")
    op.drop_index(op.f("ix_documents_access_group"), table_name="documents")
    op.drop_index(op.f("ix_documents_owner_id"), table_name="documents")
    op.drop_table("documents")
