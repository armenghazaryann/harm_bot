"""merge heads

Revision ID: aff40758a967
Revises: 79acfe87e810, 20250913_add_conversation_tables
Create Date: 2025-09-13 20:02:06.424149

"""
from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "aff40758a967"
down_revision: Union[str, None] = ("79acfe87e810", "20250913_add_conversation_tables")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
