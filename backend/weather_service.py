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


NOMINATIM_GEOCODING_URL = "https://nominatim.openstreetmap.org/search"

# Curated reference locations for instant zero-latency resolution
KNOWN_LOCATIONS: Dict[str, Dict[str, Any]] = {
    "adisaptagram": {
        "name": "Adisaptagram",
        "latitude": 22.9549,
        "longitude": 88.3790,
        "country": "India",
        "admin1": "West Bengal",
        "timezone": "Asia/Kolkata",
        "country_code": "IN",
    },
    "west bengal": {
        "name": "West Bengal",
        "latitude": 22.9965,
        "longitude": 87.6856,
        "country": "India",
        "admin1": "West Bengal",
        "timezone": "Asia/Kolkata",
        "country_code": "IN",
    },
    "hooghly": {
        "name": "Hooghly",
        "latitude": 22.8963,
        "longitude": 88.3999,
        "country": "India",
        "admin1": "West Bengal",
        "timezone": "Asia/Kolkata",
        "country_code": "IN",
    },
    "kolkata": {
        "name": "Kolkata",
        "latitude": 22.5726,
        "longitude": 88.3639,
        "country": "India",
        "admin1": "West Bengal",
        "timezone": "Asia/Kolkata",
        "country_code": "IN",
    },
    "delhi": {
        "name": "Delhi",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "country": "India",
        "admin1": "Delhi",
        "timezone": "Asia/Kolkata",
        "country_code": "IN",
    },
    "mumbai": {
        "name": "Mumbai",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "country": "India",
        "admin1": "Maharashtra",
        "timezone": "Asia/Kolkata",
        "country_code": "IN",
    },
    "bengaluru": {
        "name": "Bengaluru",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "country": "India",
        "admin1": "Karnataka",
        "timezone": "Asia/Kolkata",
        "country_code": "IN",
    },
    "chennai": {
        "name": "Chennai",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "country": "India",
        "admin1": "Tamil Nadu",
        "timezone": "Asia/Kolkata",
        "country_code": "IN",
    },
}

WEATHER_PATTERNS = [
    r"\b(?:what(?:'s|s|\s+is)\s+)?(?:the\s+)?(?:current\s+)?(?:today'?s?\s+)?(?:weather|temperature|temp|forecast|humidity|climate)\b",
    r"\bhow(?:'s|s|\s+is)\s+(?:the\s+)?weather\b",
    r"\bweather\s+(?:in|for|at|around|of)\b",
    r"\bforecast\s+(?:in|for|at|around|of)\b",
    r"\btemperature\s+(?:in|for|at|around|of)\b",
    r"\bis\s+it\s+(?:raining|hot|cold|sunny|cloudy|snowing)\s+(?:in|at|for|around)\b",
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
      'whats the weather in west bengal' -> 'west bengal'
      'How is the weather in Tokyo, Japan right now?' -> 'Tokyo, Japan'
      'forecast for New York' -> 'New York'
    """
    cleaned = query.strip()

    # Reject generic inquiries with no location specified
    if re.match(r"^(?:what(?:'s|s|\s+is)\s+)?(?:the\s+)?(?:current\s+)?(?:today'?s?\s+)?weather\??$", cleaned, re.IGNORECASE):
        return None

    # Match patterns like: "in <place>", "for <place>", "at <place>", "weather of <place>"
    patterns = [
        r"\b(?:in|for|at|around)\s+([A-Za-z0-9_\s,.-]+?)(?:\s+(?:today|tomorrow|now|right\s+now|this\s+week))?(?:\?|\.|\s*$)",
        r"\b(?:weather|forecast|temperature|temp|climate)\s+(?:of|in|for|at)\s+([A-Za-z0-9_\s,.-]+?)(?:\s+(?:today|tomorrow|now|right\s+now|this\s+week))?(?:\?|\.|\s*$)",
        r"^([A-Za-z0-9_\s,.-]+?)\s+(?:weather|forecast|temperature|temp|climate)\??$",
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


def _geocode_open_meteo(
    clean_name: str,
    client: Optional[httpx.Client] = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
) -> List[Dict[str, Any]]:
    """Queries Open-Meteo Geocoding API."""
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
        logger.debug("Open-Meteo geocoding request failed for '%s': %s", clean_name, exc)
        return []


def _geocode_nominatim(
    clean_name: str,
    client: Optional[httpx.Client] = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
) -> List[Dict[str, Any]]:
    """Queries OpenStreetMap Nominatim API for states, regions, stations, villages."""
    params = {
        "q": clean_name,
        "format": "json",
        "limit": 5,
        "addressdetails": 1,
    }
    headers = {"User-Agent": "KAIROS-WeatherService/0.6.0 (kairos-assistant@local.dev)"}
    try:
        if client is not None:
            resp = client.get(NOMINATIM_GEOCODING_URL, params=params, headers=headers, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as cl:
                resp = cl.get(NOMINATIM_GEOCODING_URL, params=params, headers=headers)

        if resp.status_code != 200:
            return []

        data = resp.json()
        if not isinstance(data, list):
            return []

        candidates: List[Dict[str, Any]] = []
        for item in data:
            lat = item.get("lat")
            lon = item.get("lon")
            if lat is None or lon is None:
                continue

            address = item.get("address", {})
            country = address.get("country", "")
            country_code = address.get("country_code", "").upper()
            admin1 = address.get("state") or address.get("state_district") or address.get("county") or ""
            name = (
                item.get("name")
                or address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("state")
                or clean_name.title()
            )

            candidates.append({
                "name": name,
                "latitude": float(lat),
                "longitude": float(lon),
                "country": country,
                "admin1": admin1,
                "timezone": "",
                "country_code": country_code,
            })
        return candidates
    except Exception as exc:
        logger.debug("Nominatim geocoding request failed for '%s': %s", clean_name, exc)
        return []


def geocode_location(
    location_name: str,
    client: Optional[httpx.Client] = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
) -> List[Dict[str, Any]]:
    """
    Queries geocoding providers to resolve a placename to geographic coordinates.
    1. Primary: Open-Meteo Geocoding API.
    2. Curated Reference Locations: instant zero-latency match (e.g. West Bengal, Adisaptagram).
    3. Fallback: OpenStreetMap Nominatim API (covers global states, districts, stations, villages).
    """
    clean_name = location_name.strip()
    if not clean_name:
        return []

    # 1. Primary: Open-Meteo Geocoding
    candidates = _geocode_open_meteo(clean_name, client=client, timeout=timeout)
    if candidates:
        return candidates

    # 2. Curated known reference locations
    key = clean_name.lower().strip()
    if key in KNOWN_LOCATIONS:
        return [dict(KNOWN_LOCATIONS[key])]

    compact_key = re.sub(r"[\s\-_]+", "", key)
    for k, loc in KNOWN_LOCATIONS.items():
        if re.sub(r"[\s\-_]+", "", k) == compact_key:
            return [dict(loc)]

    # 3. Fallback: OpenStreetMap Nominatim
    candidates = _geocode_nominatim(clean_name, client=client, timeout=timeout)
    if candidates:
        return candidates

    return []


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
    admin = (
        f", {selected['admin1']}"
        if selected.get("admin1") and selected["admin1"].strip().lower() != loc_name.strip().lower()
        else ""
    )
    country = f", {selected['country']}" if selected.get("country") else ""
    full_loc_str = f"{loc_name}{admin}{country}"

    cache_note = "\n\n*(Retrieved from recent 10-minute cache)*" if weather.get("cached") else ""

    temp_c = weather.get("temperature_c", 0.0)
    temp_f = weather.get("temperature_f", round((temp_c * 9 / 5) + 32, 1))
    apparent_c = weather.get("apparent_temperature_c", temp_c)
    apparent_f = weather.get("apparent_temperature_f", round((apparent_c * 9 / 5) + 32, 1))
    humidity = weather.get("humidity_percent", "N/A")
    wind = weather.get("wind_speed_kmh", 0.0)
    precip = weather.get("precipitation_mm", 0.0)
    tz_str = weather.get("timezone", "UTC")
    cond_desc = weather.get("condition_description", "Reported conditions")
    cond_emoji = weather.get("condition_emoji", "🌤️")

    formatted_msg = (
        f"### {cond_emoji} Weather for **{full_loc_str}**\n\n"
        f"- **Condition:** {cond_desc} {cond_emoji}\n"
        f"- **Temperature:** **{temp_c}°C** ({temp_f}°F)\n"
        f"- **Feels Like:** {apparent_c}°C ({apparent_f}°F)\n"
        f"- **Relative Humidity:** {humidity}%\n"
        f"- **Wind Speed:** {wind} km/h\n"
        f"- **Precipitation:** {precip} mm\n"
        f"- **Timezone:** {tz_str}"
        f"{cache_note}"
    )

    return {
        "status": "success",
        "message": formatted_msg,
        "location": selected,
        "weather": weather,
        "cached": weather.get("cached", False),
    }
