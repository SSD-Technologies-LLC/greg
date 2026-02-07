"""Tests for Tavily web search integration."""

from unittest.mock import MagicMock

import pytest

from src.brain.searcher import WebSearcher


@pytest.fixture
def mock_tavily():
    client = MagicMock()
    client.search = MagicMock(
        return_value={
            "answer": "Bitcoin is currently at $95,000",
            "results": [
                {"title": "BTC Price", "url": "https://example.com/btc", "content": "Bitcoin is at $95k"},
                {"title": "Crypto News", "url": "https://example.com/news", "content": "Markets are up"},
            ],
        }
    )
    return client


@pytest.fixture
def searcher(mock_tavily):
    return WebSearcher(tavily_client=mock_tavily)


class TestSearch:
    def test_basic_search(self, searcher, mock_tavily):
        result = searcher.search("Bitcoin price")
        assert result is not None
        assert "95,000" in result or "95k" in result
        mock_tavily.search.assert_called_once()

    def test_search_formats_results(self, searcher):
        result = searcher.search("test query")
        assert "https://example.com" in result

    def test_search_includes_answer(self, searcher):
        result = searcher.search("test query")
        assert "95,000" in result

    def test_search_with_no_results(self, searcher, mock_tavily):
        mock_tavily.search = MagicMock(return_value={"answer": None, "results": []})
        result = searcher.search("obscure query")
        assert result is None

    def test_search_handles_tavily_error(self, searcher, mock_tavily):
        mock_tavily.search = MagicMock(side_effect=Exception("API error"))
        result = searcher.search("test")
        assert result is None

    def test_respects_max_results(self, mock_tavily):
        mock_tavily.search = MagicMock(
            return_value={
                "answer": None,
                "results": [
                    {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": f"Content {i}"}
                    for i in range(10)
                ],
            }
        )
        searcher = WebSearcher(tavily_client=mock_tavily)
        result = searcher.search("test", max_results=2)
        assert result.count("https://example.com/") == 2


class TestSearchDisabled:
    def test_no_client_returns_none(self):
        searcher = WebSearcher(tavily_client=None)
        result = searcher.search("test")
        assert result is None
