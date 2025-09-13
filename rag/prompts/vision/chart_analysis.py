"""Vision prompts for analyzing charts and financial visualizations in PDFs.

Uses OpenAI Vision API to extract insights from charts, graphs, and tables
in earnings materials.
"""

CHART_ANALYSIS_PROMPT = """You are a financial analyst expert at interpreting charts and visualizations
from earnings materials. Analyze this image and extract key financial insights.

Focus on:
- Specific numerical values and trends
- Time periods covered
- Key metrics being displayed
- Notable changes or patterns
- Business implications

Provide a structured analysis in the following format:

**Chart Type**: [Bar chart/Line graph/Pie chart/Table/etc.]

**Key Metrics**: [List the main metrics shown]

**Time Period**: [Date range or periods covered]

**Key Insights**:
- [Insight 1 with specific numbers]
- [Insight 2 with specific numbers]
- [Additional insights]

**Notable Trends**: [Describe any significant trends or changes]

**Business Context**: [What this data suggests about the company's performance]

Be precise with numbers and avoid speculation beyond what's clearly shown in the image."""

IMAGE_DESCRIPTION_PROMPT = """Describe this image from an earnings document. Focus on:
- What type of visual element it is (chart, table, diagram, etc.)
- Key text and numerical information visible
- Overall structure and layout
- Any notable visual elements

Provide a clear, factual description that would help someone understand the content without seeing the image."""


def build_chart_analysis_prompt() -> str:
    """Build prompt for detailed chart analysis."""
    return CHART_ANALYSIS_PROMPT


def build_image_description_prompt() -> str:
    """Build prompt for general image description."""
    return IMAGE_DESCRIPTION_PROMPT
