from tavily import TavilyClient
from app.config import TAVILY_API_KEY

_tavily_client = None


def get_tavily_client():
    """Loads the Tavily web search client once and reuses it."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


def search_web(query: str, max_results: int = 4):
    """Searches the web and returns a list of {title, url, content} results.

    Returns an empty list if no API key is configured or the search fails,
    so callers can gracefully fall back to a generic message.
    """
    if not TAVILY_API_KEY:
        return []

    try:
        client = get_tavily_client()
        response = client.search(query=query, max_results=max_results)

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "content": (item.get("content") or "")[:600],
            })
        return results
    except Exception:
        return []
