"""Pluggable search backends for LongMem4LangGraph.

Two implementations ship with the library:
- SqliteLikeSearch: zero-dependency LIKE-based search (default)
- WeaviateSearch: semantic vector search via Weaviate REST API (optional)

No external Python dependencies — WeaviateSearch uses stdlib urllib.

Usage:
    from longmem4langgraph.search import SqliteLikeSearch, WeaviateSearch
    from longmem4langgraph import HistoryStore

    # Default (zero dependencies, works everywhere)
    history = HistoryStore(db_path, search_backend=SqliteLikeSearch(conn))

    # Semantic search via Weaviate
    history = HistoryStore(db_path, search_backend=WeaviateSearch(
        endpoint="http://localhost:8070",
        vectorizer="text2vec-transformers",
    ))
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from .base import SearchBackend

logger = logging.getLogger(__name__)


class SqliteLikeSearch(SearchBackend):
    """Zero-dependency search using SQLite LIKE.

    Default search backend. Works everywhere with no external services.
    Useful for quick debugging and simple text matching.

    Usage:
        search = SqliteLikeSearch(conn)
        results = search.search("dependency injection", request_id="req-1")

    Note: Special LIKE characters (% and _) are escaped to prevent
    unintended wildcard matching.
    """

    def __init__(self, conn: Any):
        self._conn = conn

    def search(self, query: str, request_id: str, limit: int = 10) -> List[dict]:
        safe = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._conn.execute(
            """SELECT * FROM agent_history
               WHERE request_id = ? AND content LIKE ?
               ORDER BY turn_number LIMIT ?""",
            (request_id, f"%{safe}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def index(self, request_id: str, source: str, content: str, **kwargs) -> None:
        pass


class WeaviateSearch(SearchBackend):
    """Semantic search via Weaviate REST API.

    Uses Weaviate's GraphQL API for vector or hybrid search.
    Falls back gracefully if Weaviate is unavailable — search()
    returns empty list, index() logs a debug warning.

    No pip dependencies required — uses urllib (stdlib).

    Usage:
        search = WeaviateSearch(
            endpoint="http://localhost:8070",
            vectorizer="text2vec-transformers",
        )
        results = search.search("dependency injection", request_id="req-1")

    Args:
        endpoint: Weaviate HTTP endpoint (default: localhost:8070).
        class_name: Weaviate class name for storing history vectors.
        vectorizer: Weaviate vectorizer module name.
            Use "text2vec-transformers" if your Weaviate has BGE-M3 or similar.
            Use "none" if you want BM25 keyword search only.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8070",
        class_name: str = "LongMemHistory",
        vectorizer: str = "none",
    ):
        self._endpoint = endpoint.rstrip("/")
        self._class_name = class_name
        self._vectorizer = vectorizer
        self._class_ready = False
        self._connection_ok = None

    def _ensure_class(self) -> None:
        if self._class_ready:
            return
        if self._connection_ok is False:
            return
        try:
            self._request(f"/v1/schema/{self._class_name}", method="GET")
            self._class_ready = True
            self._connection_ok = True
        except urllib.error.HTTPError:
            self._request("/v1/schema", data={
                "class": self._class_name,
                "vectorizer": self._vectorizer,
                "properties": [
                    {"name": "request_id", "dataType": ["text"]},
                    {"name": "source", "dataType": ["text"]},
                    {"name": "content", "dataType": ["text"]},
                    {"name": "turn_number", "dataType": ["int"]},
                    {"name": "content_type", "dataType": ["text"]},
                    {"name": "metadata", "dataType": ["text"]},
                    {"name": "thread_id", "dataType": ["text"]},
                ],
            })
            self._class_ready = True
            self._connection_ok = True
        except (ConnectionError, urllib.error.URLError) as e:
            self._connection_ok = False
            logger.warning("Weaviate unavailable at %s: %s", self._endpoint, e)

    def _request(
        self, path: str, data: Optional[dict] = None, method: str = "POST"
    ) -> dict:
        url = f"{self._endpoint}{path}"
        headers = {"Content-Type": "application/json"}
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            logger.warning("Weaviate HTTP %s: %s", e.code, error_body)
            raise
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot connect to Weaviate at {self._endpoint}: {e.reason}"
            )

    def search(self, query: str, request_id: str, limit: int = 10) -> List[dict]:
        self._ensure_class()
        if not self._class_ready:
            return []

        try:
            where_filter = {
                "operator": "And",
                "operands": [
                    {
                        "path": ["request_id"],
                        "operator": "Equal",
                        "valueText": request_id,
                    },
                    {
                        "path": ["content"],
                        "operator": "Like",
                        "valueText": f"*{query}*",
                    },
                ],
            }
            gql = """{{
                Get {{
                    {cls}(
                        limit: {limit},
                        where: {where}
                    ) {{
                        request_id
                        source
                        content
                        turn_number
                        content_type
                        metadata
                        _additional {{ certainty }}
                    }}
                }}
            }}""".format(
                cls=self._class_name,
                limit=limit,
                where=json.dumps(where_filter),
            )
            result = self._request("/v1/graphql", data={"query": gql})
            items = (
                result.get("data", {})
                .get("Get", {})
                .get(self._class_name, [])
            )
            return [
                {
                    "request_id": item.get("request_id"),
                    "source": item.get("source"),
                    "content": item.get("content"),
                    "turn_number": item.get("turn_number"),
                    "content_type": item.get("content_type"),
                    "metadata": item.get("metadata"),
                    "score": item.get("_additional", {}).get("certainty", 0),
                }
                for item in items
            ]
        except (ConnectionError, urllib.error.URLError):
            logger.warning("Weaviate search failed")
            return []

    def index(self, request_id: str, source: str, content: str, **kwargs) -> None:
        self._ensure_class()
        if not self._class_ready:
            return
        try:
            self._request(f"/v1/objects", data={
                "class": self._class_name,
                "properties": {
                    "request_id": request_id,
                    "source": source,
                    "content": content,
                    "turn_number": kwargs.get("turn_number", 0),
                    "content_type": kwargs.get("content_type", "text"),
                    "metadata": json.dumps(kwargs.get("metadata", {})),
                    "thread_id": kwargs.get("thread_id"),
                },
            })
        except (ConnectionError, urllib.error.URLError) as e:
            logger.debug("Weaviate index failed (non-fatal): %s", e)
