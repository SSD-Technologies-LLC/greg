import logging

logger = logging.getLogger(__name__)


class WebSearcher:
    def __init__(self, tavily_client) -> None:
        self._client = tavily_client

    def search(self, query: str, max_results: int = 3) -> str | None:
        if self._client is None:
            return None

        try:
            response = self._client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=True,
            )
        except Exception:
            logger.exception("Tavily search failed for query: %s", query)
            return None

        results = response.get("results", [])
        answer = response.get("answer")

        if not results and not answer:
            return None

        parts = []
        if answer:
            parts.append(answer)
        for r in results[:max_results]:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            snippet = content[:200] if content else ""
            parts.append(f"- {title}: {snippet} ({url})")

        return "\n".join(parts)
