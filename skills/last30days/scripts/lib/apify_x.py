"""Apify-hosted X search for the last30days pipeline (opt-in backend).

Runs the third-party Apify Actor ``apidojo/twitter-scraper-lite`` through
Apify's synchronous run endpoint and returns X posts with engagement counts.
Requires ``APIFY_API_TOKEN`` and an explicit ``LAST30DAYS_X_BACKEND=apify``
pin: it is paid, it runs through a third-party developer's scraper, and it is
never part of the default X chain.

Topic search only. The Phase 2 handle lanes (from-handle and mentions) are not
implemented for this backend, so each run costs a handful of Apify events
(one search query plus a small per-post charge) rather than one per lane.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from . import http, log
from .query import leading_mentions
from .relevance import token_overlap_relevance as _compute_relevance
from .xquik import expand_xquik_queries

# Apify addresses an Actor as ``username~name`` in API paths.
ACTOR_PATH = "apidojo~twitter-scraper-lite"
_BASE_URL = "https://api.apify.com/v2"

# Posts to request per query, and how many query variants to run.
DEPTH_CONFIG = {
    "quick": {"limit": 20, "queries": 1},
    "default": {"limit": 40, "queries": 2},
    "deep": {"limit": 100, "queries": 3},
}

# Hard ceiling on what one Apify call may bill, enforced by Apify itself. A
# 100-post deep query is about $0.06, so this never trips on a normal run.
MAX_CHARGE_USD = 0.10

# The Actor run is cut off after this many seconds; the HTTP wait is slightly
# longer so Apify's own timeout answers before the socket gives up.
_RUN_TIMEOUT_SECONDS = 90
_HTTP_TIMEOUT_SECONDS = 100


def _log(msg: str):
    log.source_log("Apify", msg, tty_only=False)


def expand_apify_queries(topic: str, depth: str) -> List[str]:
    """Query variants for a topic, one per Actor run (shared with Xquik)."""
    return expand_xquik_queries(topic, depth)


def _exclusive_end(to_date: str) -> str:
    """The Actor's end bound for an inclusive ``to_date`` (next day, like X's ``until:``)."""
    end = date.fromisoformat(to_date)
    if end == date.max:
        return to_date
    return (end + timedelta(days=1)).isoformat()


def search_apify(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    token: str = "",
    deadline: float | None = None,
    cancel: Any = None,
) -> Dict[str, Any]:
    """Search X through the Apify Actor.

    Args:
        topic: Research topic
        from_date: Start date (YYYY-MM-DD)
        to_date: Inclusive end date (YYYY-MM-DD)
        depth: "quick", "default", or "deep"
        token: Apify API token
        deadline: Optional shared wall-clock deadline (``time.monotonic()``
            instant) from the X backend chain. Queries past it never start.
        cancel: Optional cancellation event.

    Returns:
        Dict with an "items" list and an optional "error" string.
    """
    if not token:
        return {"items": [], "error": "No APIFY_API_TOKEN configured"}

    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["default"])
    queries = expand_apify_queries(topic, depth)[: cfg["queries"]]
    all_items: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for query_text in queries:
        if _stopped(deadline, cancel):
            _log("chain deadline reached; skipping remaining queries")
            return {"items": all_items, "error": "Apify research cancelled or timed out"}
        items, error = _run_query(
            query_text, from_date, to_date, cfg["limit"], token,
            seen_ids=seen_ids, index_offset=len(all_items),
            deadline=deadline, cancel=cancel,
        )
        if error:
            # Auth, payment and timeout failures are fatal for the source:
            # return them so the caller settles honestly instead of empty.
            return {"items": [], "error": error}
        all_items.extend(items)

    if _stopped(deadline, cancel):
        return {"items": all_items, "error": "Apify research cancelled or timed out"}
    return {"items": all_items}


def _stopped(deadline: float | None, cancel: Any) -> bool:
    if cancel is not None and cancel.is_set():
        return True
    return deadline is not None and time.monotonic() >= deadline


def _run_query(
    query: str,
    from_date: str,
    to_date: str,
    limit: int,
    token: str,
    *,
    seen_ids: set[str],
    index_offset: int,
    deadline: float | None,
    cancel: Any,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Run the Actor once and parse its dataset into items.

    Returns ``(items, fatal_error)``. Transient failures (5xx, network,
    unexpected payloads) are logged and yield ``([], None)``.
    """
    body = {
        "searchTerms": [query],
        "sort": "Top",
        "maxItems": limit,
        "start": from_date,
        "end": _exclusive_end(to_date),
    }
    params = {"maxTotalChargeUsd": MAX_CHARGE_USD, "timeout": _RUN_TIMEOUT_SECONDS}
    url = f"{_BASE_URL}/acts/{ACTOR_PATH}/run-sync-get-dataset-items"
    _log(f"Searching: {query}")
    try:
        # retries=0: the call is billed, so a transport retry could pay twice.
        response = http.post(
            url,
            body,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=_HTTP_TIMEOUT_SECONDS,
            retries=0,
            deadline_monotonic=deadline,
            cancel=cancel,
        )
    except http.HTTPError as exc:
        status = getattr(exc, "status_code", None)
        if status == 402:
            return [], "Apify usage limit reached: payment required (402)"
        if status in (401, 403):
            return [], f"Apify auth failed ({status})"
        if status == 408:
            return [], "Apify run timed out (408)"
        _log(f"HTTP error for '{query}': {exc}")
        return [], None
    except Exception as exc:
        _log(f"Error for '{query}': {exc}")
        return [], None

    if not isinstance(response, list):
        _log(f"unexpected response shape for '{query}': {type(response).__name__}")
        return [], None

    items: List[Dict[str, Any]] = []
    for tweet in response:
        if not isinstance(tweet, dict):
            continue
        # Placeholder rows ({"noResults": true}) and demo rows carry another
        # type or no author; only real posts become evidence.
        if tweet.get("type", "tweet") != "tweet":
            continue
        tweet_id = str(tweet.get("id") or "")
        if tweet_id in seen_ids:
            continue
        item = _parse_tweet(tweet, index_offset + len(items), query)
        if item:
            seen_ids.add(tweet_id)
            items.append(item)
    return items, None


def parse_apify_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract items from a ``search_apify`` result."""
    return response.get("items", [])


def _parse_tweet(
    tweet: Dict[str, Any], index: int, query: str, id_prefix: str = "AP"
) -> Dict[str, Any] | None:
    """Parse one Actor row into the standard X item format."""
    author = tweet.get("author")
    username = ""
    if isinstance(author, dict):
        username = str(author.get("userName") or "").lstrip("@")
    tweet_id = str(tweet.get("id") or "")
    if not username or not tweet_id:
        return None
    # Always rebuilt from handle + id: the citation must be an x.com URL no
    # matter what a third-party Actor put in its url fields.
    url = f"https://x.com/{username}/status/{tweet_id}"

    text = str(tweet.get("text") or tweet.get("fullText") or "").strip()[:500]
    engagement = {
        "likes": _safe_int(tweet.get("likeCount")),
        "reposts": _safe_int(tweet.get("retweetCount")),
        "replies": _safe_int(tweet.get("replyCount")),
        "quotes": _safe_int(tweet.get("quoteCount")),
        "views": _safe_int(tweet.get("viewCount")),
        "bookmarks": _safe_int(tweet.get("bookmarkCount")),
    }
    return {
        "id": f"{id_prefix}{index + 1}",
        "text": text,
        "url": url,
        "author_handle": username,
        "date": _parse_date(tweet.get("createdAt")),
        "engagement": engagement,
        "mentioned_handles": leading_mentions(text),
        "relevance": _compute_relevance(query, text) if query else 0.7,
        "why_relevant": "",
    }


def _parse_date(created_at: Any) -> str | None:
    """``YYYY-MM-DD`` from the Actor's legacy Twitter timestamp (or ISO 8601)."""
    if not created_at or not isinstance(created_at, str):
        return None
    try:
        if len(created_at) > 10 and created_at[10] == "T":
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
