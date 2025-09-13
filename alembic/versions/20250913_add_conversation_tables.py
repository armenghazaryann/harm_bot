"""Add conversation and message tables for simple history

Revision ID: 20250913_add_conversation_tables
Revises: 20250913_add_cost_event
Create Date: 2025-09-13 22:56:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20250913_add_conversation_tables"
down_revision = "20250913_add_cost_event"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create conversation table
    op.create_table(
        "conversation",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("user_id", sa.String(100), nullable=True),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    # Create message table
    op.create_table(
        "message",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),  # 'user' or 'assistant'
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("tokens", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    # Add foreign key constraint
    op.create_foreign_key(
        "fk_message_conversation_id",
        "message",
        "conversation",
        ["conversation_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Create indexes for efficient querying
    op.create_index("ix_conversation_user_id", "conversation", ["user_id"])
    op.create_index("ix_conversation_created_at", "conversation", ["created_at"])
    op.create_index("ix_message_conversation_id", "message", ["conversation_id"])
    op.create_index("ix_message_created_at", "message", ["created_at"])
    op.create_index("ix_message_role", "message", ["role"])


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_message_role", table_name="message")
    op.drop_index("ix_message_created_at", table_name="message")
    op.drop_index("ix_message_conversation_id", table_name="message")
    op.drop_index("ix_conversation_created_at", table_name="conversation")
    op.drop_index("ix_conversation_user_id", table_name="conversation")

    # Drop foreign key constraint
    op.drop_constraint("fk_message_conversation_id", "message", type_="foreignkey")

    # Drop tables
    op.drop_table("message")
    op.drop_table("conversation")
