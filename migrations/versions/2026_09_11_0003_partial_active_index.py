"""partial active-documents index

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_documents_owner_active",
        "documents",
        ["owner_id"],
        unique=False,
        postgresql_where=sa.text("deleted = false"),
    )


def downgrade() -> None:
    op.drop_index("ix_documents_owner_active", table_name="documents")
