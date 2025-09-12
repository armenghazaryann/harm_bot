"""
Add utterance table for transcript turns (MVP)

Revision ID: 20250912_add_utterance
Revises: 65e9079d129e
Create Date: 2025-09-12 18:30:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20250912_add_utterance"
down_revision: Union[str, None] = "65e9079d129e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "utterance",
        sa.Column("document_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("utterance_id", sa.String(length=64), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("speaker", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=True),
        sa.Column("section", sa.String(length=50), nullable=True),
        sa.Column("speech", sa.Text(), nullable=False),
        sa.Column("page_spans", sa.JSON(), nullable=True),
        sa.Column("extraction_method", sa.String(length=30), nullable=True),
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["document.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("utterance_id"),
    )
    op.create_index(
        "idx_utterance_document_turn",
        "utterance",
        ["document_id", "turn_index"],
        unique=False,
    )
    op.create_index("idx_utterance_speaker", "utterance", ["speaker"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_utterance_speaker", table_name="utterance")
    op.drop_index("idx_utterance_document_turn", table_name="utterance")
    op.drop_table("utterance")
