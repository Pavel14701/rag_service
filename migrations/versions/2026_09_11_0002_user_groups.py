"""add user_groups table

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_groups",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("group_name", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("user_id", "group_name"),
    )


def downgrade() -> None:
    op.drop_table("user_groups")