"""
Lean QueryService stub retained only for suggestions.

All search/answer functionality has moved to `workers/langchain_query_service.py`.
This module stays to satisfy DI wiring and provide lightweight suggestions.
"""
import logging
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from api.features.query.exceptions import SearchError

logger = logging.getLogger("rag.query.service")


class QueryService:
    """Minimal QueryService: only suggestions are handled here."""

    async def get_query_suggestions(
        self, prefix: str = "", limit: int = 10, db_session: AsyncSession = None
    ) -> List[str]:
        """Get query suggestions based on prefix and popular queries."""
        try:
            suggestions = [
                "What was the revenue in Q1?",
                "How did the company perform compared to last year?",
                "What are the key risks mentioned?",
                "What is the guidance for next quarter?",
                "What were the main highlights?",
                "How much cash does the company have?",
                "What are the growth drivers?",
                "What challenges does the company face?",
            ]

            if prefix:
                suggestions = [
                    s for s in suggestions if s.lower().startswith(prefix.lower())
                ]

            return suggestions[:limit]
        except Exception as e:
            logger.error(f"Failed to get query suggestions: {str(e)}")
            raise SearchError(f"Failed to get suggestions: {str(e)}")
