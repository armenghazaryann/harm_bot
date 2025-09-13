"""Add cost_event table for minimal cost tracking

Revision ID: 20250913_add_cost_event
Revises: 20250912_add_utterance
Create Date: 2025-09-13 22:55:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20250913_add_cost_event"
down_revision = "20250912_add_utterance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create cost_event table
    op.create_table(
        "cost_event",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "ts",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("route", sa.String(200), nullable=False),
        sa.Column("request_id", sa.String(100), nullable=False),
        sa.Column("correlation_id", sa.String(100), nullable=True),
        sa.Column("prompt_tokens", sa.Integer, nullable=True),
        sa.Column("completion_tokens", sa.Integer, nullable=True),
        sa.Column("total_tokens", sa.Integer, nullable=True),
        sa.Column("unit_price_in_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("unit_price_out_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("cost_in_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("cost_out_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("cost_total_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="ok"),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
    )

    # Create indexes for efficient querying
    op.create_index("ix_cost_event_ts", "cost_event", ["ts"])
    op.create_index("ix_cost_event_provider_model", "cost_event", ["provider", "model"])
    op.create_index("ix_cost_event_route", "cost_event", ["route"])
    op.create_index("ix_cost_event_request_id", "cost_event", ["request_id"])
    op.create_index("ix_cost_event_status", "cost_event", ["status"])


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_cost_event_status", table_name="cost_event")
    op.drop_index("ix_cost_event_request_id", table_name="cost_event")
    op.drop_index("ix_cost_event_route", table_name="cost_event")
    op.drop_index("ix_cost_event_provider_model", table_name="cost_event")
    op.drop_index("ix_cost_event_ts", table_name="cost_event")

    # Drop table
    op.drop_table("cost_event")
