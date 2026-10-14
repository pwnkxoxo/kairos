"""
KAIROS Weather Service
Provides:
  1. Detection and location extraction for weather inquiries.
  2. Geocoding resolution via Open-Meteo Geocoding API (free, open, no API key).
  3. Real-time conditions and forecasts via Open-Meteo Forecast API.
  4. WMO code interpretation to clear, human-readable weather descriptions.
  5. In-memory caching with 10-minute TTL to respect provider rate limits.
  6. Disambiguation handling for ambiguous or multi-match locations.
"""

from datetime import datetime, timezone
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("kairos.weather")

OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CACHE_TTL_SECONDS = 600  # 10 minutes cache freshness
DEFAULT_HTTP_TIMEOUT = 8.0  # seconds

# In-memory weather cache: (rounded_lat, rounded_lon) -> (timestamp, weather_dict)
_WEATHER_CACHE: Dict[Tuple[float, float], Tuple[float, Dict[str, Any]]] = {}

# WMO Weather interpretation codes (WW)
WMO_WEATHER_CODES: Dict[int, Tuple[str, str]] = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Foggy", "🌫️"),
    48: ("Depositing rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Moderate drizzle", "🌦️"),
    55: ("Dense drizzle", "🌧️"),
    56: ("Light freezing drizzle", "🌧️❄️"),
    57: ("Dense freezing drizzle", "🌧️❄️"),
    61: ("Slight rain", "🌧️"),
    63: ("Moderate rain", "🌧️"),
    65: ("Heavy rain", "🌧️🌧️"),
    66: ("Light freezing rain", "🌧️❄️"),
    67: ("Heavy freezing rain", "🌧️❄️"),
    71: ("Slight snowfall", "🌨️"),
    73: ("Moderate snowfall", "🌨️"),
    75: ("Heavy snowfall", "❄️❄️"),
    77: ("Snow grains", "❄️"),
    80: ("Slight rain showers", "🌦️"),
    81: ("Moderate rain showers", "🌧️"),
    82: ("Violent rain showers", "⛈️"),
    85: ("Slight snow showers", "🌨️"),
    86: ("Heavy snow showers", "❄️❄️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with slight hail", "⛈️🌨️"),
    99: ("Thunderstorm with heavy hail", "⛈️🌨️"),
}


WEATHER_PATTERNS = [
    r"\b(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:current\s+)?(?:today'?s?\s+)?(?:weather|temperature|temp|forecast|humidity|climate)\b",
    r"\bhow(?:'s|\s+is)\s+(?:the\s+)?weather\b",
    r"\bweather\s+(?:in|for|at|around)\b",
    r"\bforecast\s+(?:in|for|at|around)\b",
    r"\btemperature\s+(?:in|for|at|around)\b",
    r"\bis\s+it\s+(?:raining|hot|cold|sunny|cloudy|snowing)\s+(?:in|at|for)\b",
]

_WEATHER_REGEXES = [re.compile(p, re.IGNORECASE) for p in WEATHER_PATTERNS]


def is_weather_query(message: str) -> bool:
    """
    Returns True if user message is inquiring about weather, temperature, or forecast.
    """
    cleaned = message.strip().lower()
    if not cleaned:
        return False
    for rgx in _WEATHER_REGEXES:
        if rgx.search(cleaned):
            return True
    return False


def extract_location_from_query(query: str) -> Optional[str]:
    """
    Extracts the candidate geographic location from a natural-language weather query.
    Examples:
      'What is today's weather in Adisaptagram?' -> 'Adisaptagram'
      'Weather for Kolkata' -> 'Kolkata'
      'How is the weather in Tokyo, Japan right now?' -> 'Tokyo, Japan'
      'forecast for New York' -> 'New York'
    """
    cleaned = query.strip()

    # Reject generic inquiries with no location specified
    if re.match(r"^(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:current\s+)?(?:today'?s?\s+)?weather\??$", cleaned, re.IGNORECASE):
        return None

    # Match patterns like: "in <place>", "for <place>", "at <place>", "around <place>"
    patterns = [
        r"\b(?:in|for|at|around)\s+([A-Za-z\s,.-]+?)(?:\s+(?:today|tomorrow|now|right\s+now|this\s+week|\?|$))",
        r"\b(?:weather|forecast|temperature|temp)\s+(?:of|in|for|at)\s+([A-Za-z\s,.-]+?)(?:\?|$)",
        r"^([A-Za-z\s,.-]+?)\s+weather\??$",
    ]

    for pat in patterns:
        m = re.search(pat, cleaned, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip("?.!, ")
            cand_lower = candidate.lower()
            # Disallow common question starts or non-locations
            if any(cand_lower.startswith(prefix) for prefix in ["what", "how", "tell", "show", "is it", "which"]):
                continue
            if cand_lower not in {"today", "tomorrow", "now", "here", "the world", "detail", "celsius", "fahrenheit", "the", "current", "local"}:
                return candidate

    return None


def interpret_wmo_code(code: int) -> Tuple[str, str]:
    """Returns (description, emoji) for a given WMO weather code."""
    return WMO_WEATHER_CODES.get(code, ("Unknown conditions", "🌡️"))


class WeatherServiceError(Exception):
    """Base exception for weather service errors."""
    pass


def geocode_location(
    location_name: str,
    client: Optional[httpx.Client] = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
) -> List[Dict[str, Any]]:
    """
    Queries the Open-Meteo Geocoding API to resolve a placename to geographic coordinates.
    Returns list of candidate location dictionaries.
    """
    clean_name = location_name.strip()
    if not clean_name:
        return []

    params = {
        "name": clean_name,
        "count": 5,
        "language": "en",
        "format": "json",
    }

    headers = {"User-Agent": "KAIROS-WeatherService/0.6.0"}

    try:
        if client is not None:
            resp = client.get(OPEN_METEO_GEOCODING_URL, params=params, headers=headers, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as cl:
                resp = cl.get(OPEN_METEO_GEOCODING_URL, params=params, headers=headers)

        if resp.status_code != 200:
            logger.warning("Open-Meteo Geocoding returned status %s: %s", resp.status_code, resp.text[:200])
            return []

        data = resp.json()
        results = data.get("results", [])
        candidates: List[Dict[str, Any]] = []

        for item in results:
            name = item.get("name", "")
            lat = item.get("latitude")
            lon = item.get("longitude")
            country = item.get("country", "")
            admin1 = item.get("admin1", "")
            timezone_str = item.get("timezone", "")
            country_code = item.get("country_code", "")

            if lat is not None and lon is not None:
                candidates.append({
                    "name": name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "country": country,
                    "admin1": admin1,
                    "timezone": timezone_str,
                    "country_code": country_code,
                })

        return candidates

    except Exception as exc:
        logger.error("Geocoding request failed for '%s': %s", clean_name, exc)
        raise WeatherServiceError(f"Geocoding service unavailable: {str(exc)[:150]}")


def fetch_open_meteo_weather(
    lat: float,
    lon: float,
    client: Optional[httpx.Client] = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
) -> Dict[str, Any]:
    """
    Fetches current weather observations from Open-Meteo Forecast API.
    Caches results for 10 minutes per coordinate pair (rounded to 2 decimal places).
    """
    cache_key = (round(lat, 2), round(lon, 2))
    now_ts = time.time()

    # Check in-memory cache
    if cache_key in _WEATHER_CACHE:
        cached_ts, cached_data = _WEATHER_CACHE[cache_key]
        if now_ts - cached_ts < WEATHER_CACHE_TTL_SECONDS:
            res = dict(cached_data)
            res["cached"] = True
            return res

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    headers = {"User-Agent": "KAIROS-WeatherService/0.6.0"}

    try:
        if client is not None:
            resp = client.get(OPEN_METEO_FORECAST_URL, params=params, headers=headers, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as cl:
                resp = cl.get(OPEN_METEO_FORECAST_URL, params=params, headers=headers)

        if resp.status_code != 200:
            raise WeatherServiceError(f"Open-Meteo Forecast API returned status {resp.status_code}")

        data = resp.json()
        current = data.get("current", {})
        current_units = data.get("current_units", {})
        tz_name = data.get("timezone", "UTC")

        weather_code = int(current.get("weather_code", 0))
        condition_desc, condition_emoji = interpret_wmo_code(weather_code)

        temp_c = float(current.get("temperature_2m", 0.0))
        temp_f = round((temp_c * 9 / 5) + 32, 1)
        apparent_temp_c = float(current.get("apparent_temperature", temp_c))
        apparent_temp_f = round((apparent_temp_c * 9 / 5) + 32, 1)

        result = {
            "temperature_c": temp_c,
            "temperature_f": temp_f,
            "apparent_temperature_c": apparent_temp_c,
            "apparent_temperature_f": apparent_temp_f,
            "humidity_percent": int(current.get("relative_humidity_2m", 0)),
            "precipitation_mm": float(current.get("precipitation", 0.0)),
            "wind_speed_kmh": float(current.get("wind_speed_10m", 0.0)),
            "weather_code": weather_code,
            "condition_description": condition_desc,
            "condition_emoji": condition_emoji,
            "is_day": bool(current.get("is_day", 1)),
            "observation_time": current.get("time", ""),
            "timezone": tz_name,
            "cached": False,
        }

        # Cache result
        _WEATHER_CACHE[cache_key] = (now_ts, result)
        return result

    except WeatherServiceError:
        raise
    except Exception as exc:
        logger.error("Weather forecast request failed for (%s, %s): %s", lat, lon, exc)
        raise WeatherServiceError(f"Weather forecast service unavailable: {str(exc)[:150]}")


def clear_weather_cache() -> None:
    """Clears the in-memory weather cache (used in testing)."""
    _WEATHER_CACHE.clear()


def handle_weather_query(
    user_message: str,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """
    Main entry point for handling weather inquiries:
    1. Extracts requested location.
    2. Resolves coordinates via Open-Meteo geocoding.
    3. Handles ambiguity or missing location.
    4. Fetches current weather observations with units and condition translation.
    5. Returns formatted markdown message with structured metadata.
    """
    loc_candidate = extract_location_from_query(user_message)
    if not loc_candidate:
        msg = (
            "Please specify the city or place you would like the weather for.\n\n"
            "**Examples:**\n"
            "- *What is today's weather in Adisaptagram?*\n"
            "- *Weather in Kolkata*\n"
            "- *What is the temperature in London?*"
        )
        return {
            "status": "clarification_needed",
            "message": msg,
            "location": None,
            "weather": None,
            "cached": False,
        }

    try:
        candidates = geocode_location(loc_candidate, client=client)
    except WeatherServiceError as exc:
        return {
            "status": "error",
            "message": f"Could not retrieve location data: {exc}",
            "location": None,
            "weather": None,
            "cached": False,
        }

    if not candidates:
        msg = (
            f"Could not locate **'{loc_candidate}'** on the map. "
            "Please check the spelling or provide the state or country (e.g., 'Springfield, IL' or 'Paris, France')."
        )
        return {
            "status": "clarification_needed",
            "message": msg,
            "location": None,
            "weather": None,
            "cached": False,
        }

    # Disambiguation check: if multiple matches have completely different regions (admin1 or country) and user gave a single term
    if len(candidates) >= 2:
        distinct_regions = {f"{c.get('admin1', '')}, {c.get('country', '')}".strip(', ') for c in candidates[:3]}
        if len(distinct_regions) > 1 and len(loc_candidate.split(",")) == 1:
            options_text = []
            for idx, c in enumerate(candidates[:4], start=1):
                admin = f", {c['admin1']}" if c.get('admin1') else ""
                cntry = f", {c['country']}" if c.get('country') else ""
                options_text.append(f"{idx}. **{c['name']}**{admin}{cntry}")
            first_admin = f", {candidates[0]['admin1']}" if candidates[0].get("admin1") else ""
            msg = (
                f"Multiple locations found matching **'{loc_candidate}'**:\n\n"
                + "\n".join(options_text)
                + f"\n\nPlease specify which one you meant (for example: *'Weather in {candidates[0]['name']}{first_admin}'*)."
            )
            return {
                "status": "clarification_needed",
                "message": msg,
                "location": None,
                "weather": None,
                "cached": False,
            }

    selected = candidates[0]
    lat = selected["latitude"]
    lon = selected["longitude"]

    try:
        weather = fetch_open_meteo_weather(lat, lon, client=client)
    except WeatherServiceError as exc:
        return {
            "status": "error",
            "message": f"Could not retrieve weather for **{selected['name']}**: {exc}",
            "location": selected,
            "weather": None,
            "cached": False,
        }

    loc_name = selected["name"]
    admin = f", {selected['admin1']}" if selected.get("admin1") else ""
    country = f", {selected['country']}" if selected.get("country") else ""
    full_loc_str = f"{loc_name}{admin}{country}"

    cache_note = "\n\n*(Retrieved from recent 10-minute cache)*" if weather.get("cached") else ""

    formatted_msg = (
        f"### {weather['condition_emoji']} Weather for **{full_loc_str}**\n\n"
        f"- **Condition:** {weather['condition_description']} {weather['condition_emoji']}\n"
        f"- **Temperature:** **{weather['temperature_c']}°C** ({weather['temperature_f']}°F)\n"
        f"- **Feels Like:** {weather['apparent_temperature_c']}°C ({weather['apparent_temperature_f']}°F)\n"
        f"- **Relative Humidity:** {weather['humidity_percent']}%\n"
        f"- **Wind Speed:** {weather['wind_speed_kmh']} km/h\n"
        f"- **Precipitation:** {weather['precipitation_mm']} mm\n"
        f"- **Timezone:** {weather['timezone']}"
        f"{cache_note}"
    )

    return {
        "status": "success",
        "message": formatted_msg,
        "location": selected,
        "weather": weather,
        "cached": weather.get("cached", False),
    }
