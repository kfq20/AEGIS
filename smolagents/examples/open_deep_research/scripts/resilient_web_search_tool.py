from __future__ import annotations

from typing import Optional

from smolagents import Tool


class ResilientWebSearchTool(Tool):
    """A resilient web search tool that tries Serper first, then falls back to SerpAPI.

    - Name is kept as 'web_search' so existing prompts use it naturally.
    - Never raises on empty results; returns a helpful string instead.
    """

    name = "web_search"
    description = "Perform a web search via Serper (fallback to SerpAPI) and return top results as a formatted string."
    inputs = {
        "query": {"type": "string", "description": "The web search query to perform."},
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self) -> None:
        super().__init__()
        # Lazy import to avoid import-time env checks raising
        from smolagents.default_tools import GoogleSearchTool  # noqa
        self._serper = GoogleSearchTool(provider="serper")
        try:
            self._serpapi = GoogleSearchTool(provider="serpapi")
        except Exception:
            self._serpapi = None

    def forward(self, query: str, filter_year: Optional[int] = None) -> str:
        # Try Serper first
        try:
            return self._serper.forward(query=query, filter_year=filter_year)
        except Exception as e1:  # noqa: BLE001
            # Fallback to SerpAPI when available
            if self._serpapi is not None:
                try:
                    return self._serpapi.forward(query=query, filter_year=filter_year)
                except Exception as e2:  # noqa: BLE001
                    return (
                        f"No results found for query: '{query}'. "
                        f"Serper error: {e1!r}; SerpAPI error: {e2!r}. "
                        "Try a broader query or remove date filters."
                    )
            return (
                f"No results found for query: '{query}'. Serper error: {e1!r}. "
                "Try a broader query or remove date filters."
            )

