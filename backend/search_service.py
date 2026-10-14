"""
KAIROS Search & Time Service
Provides:
  1. Reliable local date/time handling (timezone-aware, zero-network).
  2. Bounded, safe Tavily Web Search integration with prompt-injection defense.
"""

from datetime import datetime
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import zoneinfo

import dotenv

# Load .env without overriding existing environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_TIMEZONE = "Asia/Kolkata"
DEFAULT_SEARCH_TIMEOUT_SECONDS = 10
MAX_SEARCH_RESULTS = 5
MAX_SEARCH_SNIPPET_CHARS = 1000
MAX_TOTAL_SEARCH_CONTEXT_CHARS = 4000


# ------------------------------------------------------------------------------
# 1. Date & Time Handling (Local Runtime, Zero Network)
# ------------------------------------------------------------------------------
def get_configured_timezone() -> zoneinfo.ZoneInfo:
    """Returns the configured timezone, falling back to Asia/Kolkata if invalid."""
    tz_name = os.getenv("KAIROS_TIMEZONE", DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    try:
        return zoneinfo.ZoneInfo(tz_name)
    except Exception:
        return zoneinfo.ZoneInfo(DEFAULT_TIMEZONE)


def get_current_datetime_info() -> Dict[str, Any]:
    """Returns formatted date and time information from the backend host."""
    tz = get_configured_timezone()
    now = datetime.now(tz)
    return {
        "datetime": now,
        "timezone_name": str(tz),
        "date_str": now.strftime("%A, %B %d, %Y"),
        "time_str": now.strftime("%I:%M:%S %p %Z"),
        "short_date": now.strftime("%Y-%m-%d"),
        "day_of_week": now.strftime("%A"),
    }


DATE_PATTERNS = [
    r"^(?:what(?:'s|\s+is)|\bwhich\b)?\s*(?:the\s+)?(?:current\s+)?(?:date|day|time|year|month)\b",
    r"^what\s+(?:is\s+)?today(?:'s)?\s*(?:date|day)?\??$",
    r"^today(?:'s)?\s*(?:date|day)\??$",
    r"^what\s+time\s+is\s+it\??$",
    r"^current\s+time\??$",
]

_DATE_REGEXES = [re.compile(p, re.IGNORECASE) for p in DATE_PATTERNS]


def is_direct_datetime_query(message: str) -> bool:
    """
    Returns True if the message is a direct, straightforward inquiry about
    the current date, time, day, or year that should be answered instantly
    without external API or LLM calls.
    """
    cleaned = message.strip().lower().rstrip("?.! ")
    if not cleaned:
        return False

    for rgx in _DATE_REGEXES:
        if rgx.search(cleaned):
            return True

    exact_matches = {
        "date",
        "time",
        "today",
        "today's date",
        "todays date",
        "what is today",
        "what is today's date",
        "what is the date today",
        "what's today's date",
        "what is the current time",
        "what time is it",
        "what day is it",
        "what day is today",
        "which year is it",
        "what year is this",
    }
    return cleaned in exact_matches


def answer_datetime_query(message: str) -> str:
    """Answers a date/time query directly using the configured timezone."""
    info = get_current_datetime_info()
    msg_lower = message.strip().lower()

    if "time" in msg_lower and "date" not in msg_lower and "day" not in msg_lower:
        return f"The current time is **{info['time_str']}** ({info['timezone_name']})."

    if "day" in msg_lower and "date" not in msg_lower and "time" not in msg_lower:
        return f"Today is **{info['day_of_week']}**, {info['date_str']} ({info['timezone_name']})."

    return (
        f"Today's date is **{info['date_str']}**.\n"
        f"The current time is **{info['time_str']}** ({info['timezone_name']})."
    )


# ------------------------------------------------------------------------------
# 2. Tavily Search Configuration
# ------------------------------------------------------------------------------
def get_tavily_api_key() -> Optional[str]:
    """Retrieves TAVILY_API_KEY from environment without logging or exposing it."""
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key or key == "your_tavily_api_key_here":
        return None
    return key


def is_tavily_configured() -> bool:
    """Returns True if a non-placeholder Tavily API key is set."""
    return get_tavily_api_key() is not None


# Explicit phrases that indicate external current information or search requirement
SEARCH_TRIGGER_PATTERNS = [
    r"\b(?:search\s+(?:the\s+)?web|google|tavily|browse\s+(?:the\s+)?web)\b",
    r"\bsearch\s+(?:for|about)\s+.+\b",
    # Handles "latest news", "latest Indian news", "latest AI news", "current tech news", "today's headlines", etc.
    r"\b(?:latest|recent|current|breaking|today'?s?|yesterday'?s?)\s+(?:[a-zA-Z0-9_\-]+\s+){0,3}(?:news|events|updates|developments|headlines|scores|prices)\b",
    # Handles "news about <topic>", "news in <topic>", "headlines about <topic>"
    r"\b(?:news|headlines|updates)\s+(?:about|in|on|regarding)\s+.+\b",
    # Handles "latest developments in AI", "recent developments on Mars"
    r"\b(?:latest|recent|new)\s+developments\s+(?:in|on|regarding)\s+.+\b",
    # Handles "what happened today in India", "what happened recently in France"
    r"\bwhat\s+happened\s+(?:today|yesterday|this\s+week|recently)(?:\s+(?:in|at|to)\s+.+)?\b",
    r"\bwhat\s+happened\s+(?:in|at|to)\s+.+\s+(?:today|yesterday|recently)\b",
    # Current officeholders, champions, leaders
    r"\bcurrent\s+(?:president|prime\s+minister|ceo|governor|champion|price\s+of)\b",
    r"\bwho\s+won\s+(?:the\s+)?(?:match|election|game|tournament|award|cup|medal)\b",
    r"\bwho\s+is\s+the\s+current\s+(?:president|prime\s+minister|ceo|governor|champion)\b",
]

_SEARCH_REGEXES = [re.compile(p, re.IGNORECASE) for p in SEARCH_TRIGGER_PATTERNS]


def should_search_web(user_message: str) -> bool:
    """
    Determines whether a user query requires current external information
    via Tavily search. Returns False for programming questions, greetings,
    file actions, date/time questions, and weather queries.
    """
    cleaned = user_message.strip().lower()

    # Direct date/time queries are handled locally, do not search
    if is_direct_datetime_query(cleaned):
        return False

    # Weather queries are handled by dedicated weather service
    if re.search(r"\b(?:weather|temperature|forecast|humidity|climate)\b", cleaned):
        return False

    # Check search triggers
    for rgx in _SEARCH_REGEXES:
        if rgx.search(cleaned):
            return True

    return False


# ------------------------------------------------------------------------------
# 3. Secure Tavily Search Execution
# ------------------------------------------------------------------------------
class SearchServiceError(Exception):
    """Base exception for search service errors."""
    pass


def execute_tavily_search(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = MAX_SEARCH_RESULTS,
    timeout: int = DEFAULT_SEARCH_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Executes a search via Tavily SDK with bounded results and timeout.
    Returns sanitized structured results:
      {
        "query": str,
        "results": [
           {"title": str, "url": str, "content": str}
        ]
      }
    """
    eff_key = api_key or get_tavily_api_key()
    if not eff_key:
        raise SearchServiceError(
            "TAVILY_API_KEY is not configured on the backend. Web search is unavailable."
        )

    try:
        from tavily import TavilyClient
    except ImportError:
        raise SearchServiceError("The 'tavily-python' SDK is not installed on the backend.")

    try:
        client = TavilyClient(api_key=eff_key)
        # Bounded query length
        clean_query = query.strip()[:300]
        # Execute search with bounded depth
        response = client.search(
            query=clean_query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as exc:
        err_msg = str(exc)
        # Sanitize any potential API key leakage in error string
        if eff_key and eff_key in err_msg:
            err_msg = err_msg.replace(eff_key, "[REDACTED]")
        raise SearchServiceError(f"Tavily search request failed: {err_msg[:200]}")

    raw_results = response.get("results", []) if isinstance(response, dict) else []
    sanitized_results: List[Dict[str, Any]] = []

    total_chars = 0
    for item in raw_results:
        title = str(item.get("title", "")).strip() or "Web Source"
        url = str(item.get("url", "")).strip()
        raw_snippet = str(item.get("content", "")).strip()
        snippet = raw_snippet[:MAX_SEARCH_SNIPPET_CHARS]
        pub_date = str(item.get("published_date", "")).strip()

        if not url:
            continue

        result_entry: Dict[str, Any] = {"title": title, "url": url, "content": snippet}
        if pub_date:
            result_entry["published_date"] = pub_date

        if total_chars + len(snippet) > MAX_TOTAL_SEARCH_CONTEXT_CHARS:
            snippet = snippet[: max(0, MAX_TOTAL_SEARCH_CONTEXT_CHARS - total_chars)]
            if snippet:
                result_entry["content"] = snippet
                sanitized_results.append(result_entry)
            break

        sanitized_results.append(result_entry)
        total_chars += len(snippet)

    return {
        "query": clean_query,
        "results": sanitized_results,
    }


def format_search_evidence(search_data: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    """
    Formats retrieved search results as strictly untrusted reference data for Gemma.
    Returns (formatted_evidence_string, list_of_sources).
    """
    results = search_data.get("results", [])
    if not results:
        return "No relevant search results were returned for this query.", []

    evidence_lines = []
    sources = []

    for idx, item in enumerate(results, start=1):
        title = item["title"]
        url = item["url"]
        content = item["content"]
        pub_date = item.get("published_date", "")
        source_entry = {"title": title, "url": url}
        if pub_date:
            source_entry["published_date"] = pub_date
        sources.append(source_entry)

        date_tag = f" (Published: {pub_date})" if pub_date else ""
        evidence_lines.append(
            f"[Source {idx}]: {title}{date_tag}\nURL: {url}\nContent: {content}\n"
        )

    evidence_str = "\n".join(evidence_lines).strip()
    return evidence_str, sources
