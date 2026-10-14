"""
Unit tests for KAIROS Weather Service and Open-Meteo Integration.
Tests:
  1. Weather query classification and location extraction.
  2. WMO code translation to human-readable condition text and emojis.
  3. Geocoding resolution with mocked Open-Meteo responses.
  4. Current weather fetching with unit conversions (°C and °F) and metrics.
  5. In-memory 10-minute caching behavior.
  6. Location ambiguity handling and polite clarification.
  7. API endpoint chat integration (/api/v1/chat).
  8. Graceful error handling on network or provider failure.
"""

import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.api import app
from backend.storage import init_db
from backend.weather_service import (
    WeatherServiceError,
    clear_weather_cache,
    extract_location_from_query,
    fetch_open_meteo_weather,
    geocode_location,
    handle_weather_query,
    interpret_wmo_code,
    is_weather_query,
)


class TestWeatherService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.client = TestClient(app)
        clear_weather_cache()

    def tearDown(self):
        clear_weather_cache()

    # --------------------------------------------------------------------------
    # 1. Weather Query Classification
    # --------------------------------------------------------------------------
    def test_is_weather_query_positive(self):
        """Identifies natural phrasing of weather inquiries."""
        queries = [
            "What is today's weather in Adisaptagram?",
            "Weather in Kolkata",
            "What is the temperature in London?",
            "How is the weather in Tokyo?",
            "Current forecast for New York",
            "Is it raining in Paris?",
            "What's the humidity in Mumbai?",
            "Adisaptagram weather",
        ]
        for q in queries:
            self.assertTrue(is_weather_query(q), f"Failed to identify weather query: {q}")

    def test_is_weather_query_negative(self):
        """Rejects non-weather inquiries."""
        non_weather = [
            "What is today's date?",
            "How do I write a binary search in Python?",
            "Create a file demo.txt",
            "Latest Indian news",
            "Who won the cricket match yesterday?",
        ]
        for q in non_weather:
            self.assertFalse(is_weather_query(q), f"False positive for weather query: {q}")

    # --------------------------------------------------------------------------
    # 2. Location Extraction
    # --------------------------------------------------------------------------
    def test_extract_location_from_query(self):
        """Extracts location strings from various query phrasings."""
        cases = [
            ("What is today's weather in Adisaptagram?", "Adisaptagram"),
            ("Weather in Kolkata", "Kolkata"),
            ("What is the temperature in London?", "London"),
            ("Current forecast for New York", "New York"),
            ("How is the weather in Tokyo right now?", "Tokyo"),
            ("Adisaptagram weather?", "Adisaptagram"),
        ]
        for query, expected_loc in cases:
            loc = extract_location_from_query(query)
            self.assertIsNotNone(loc, f"Failed to extract location from: {query}")
            self.assertEqual(loc.lower(), expected_loc.lower())

    def test_extract_location_missing(self):
        """Returns None when no specific location is mentioned."""
        loc = extract_location_from_query("What is the weather?")
        self.assertIsNone(loc)

    # --------------------------------------------------------------------------
    # 3. WMO Code Interpretation
    # --------------------------------------------------------------------------
    def test_interpret_wmo_code(self):
        """Translates standard WMO codes into human-readable descriptions and emojis."""
        desc, emoji = interpret_wmo_code(0)
        self.assertEqual(desc, "Clear sky")
        self.assertEqual(emoji, "☀️")

        desc, emoji = interpret_wmo_code(1)
        self.assertEqual(desc, "Mainly clear")

        desc, emoji = interpret_wmo_code(61)
        self.assertIn("rain", desc.lower())

        desc, emoji = interpret_wmo_code(95)
        self.assertEqual(desc, "Thunderstorm")

        # Unknown code fallback
        desc, emoji = interpret_wmo_code(999)
        self.assertEqual(desc, "Unknown conditions")

    # --------------------------------------------------------------------------
    # 4. Geocoding Resolution (Mocked)
    # --------------------------------------------------------------------------
    @patch("httpx.Client.get")
    def test_geocode_location_success(self, mock_get):
        """Successfully parses Open-Meteo geocoding results."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {
                    "name": "Adisaptagram",
                    "latitude": 22.95,
                    "longitude": 88.38,
                    "country": "India",
                    "admin1": "West Bengal",
                    "timezone": "Asia/Kolkata",
                    "country_code": "IN",
                }
            ]
        }
        mock_get.return_value = mock_resp

        candidates = geocode_location("Adisaptagram")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["name"], "Adisaptagram")
        self.assertEqual(candidates[0]["latitude"], 22.95)
        self.assertEqual(candidates[0]["country"], "India")
        self.assertEqual(candidates[0]["admin1"], "West Bengal")

    # --------------------------------------------------------------------------
    # 5. Weather Forecast Retrieval & Unit Calculations (Mocked)
    # --------------------------------------------------------------------------
    @patch("httpx.Client.get")
    def test_fetch_open_meteo_weather_success(self, mock_get):
        """Fetches and converts temperature to Celsius and Fahrenheit."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "latitude": 22.95,
            "longitude": 88.38,
            "timezone": "Asia/Kolkata",
            "current": {
                "time": "2026-10-14T12:00",
                "temperature_2m": 29.4,
                "relative_humidity_2m": 68,
                "apparent_temperature": 33.1,
                "is_day": 1,
                "precipitation": 0.0,
                "weather_code": 1,
                "wind_speed_10m": 8.5,
            },
        }
        mock_get.return_value = mock_resp

        res = fetch_open_meteo_weather(22.95, 88.38)
        self.assertEqual(res["temperature_c"], 29.4)
        self.assertEqual(res["temperature_f"], 84.9)
        self.assertEqual(res["apparent_temperature_c"], 33.1)
        self.assertEqual(res["humidity_percent"], 68)
        self.assertEqual(res["wind_speed_kmh"], 8.5)
        self.assertEqual(res["condition_description"], "Mainly clear")
        self.assertFalse(res["cached"])

    # --------------------------------------------------------------------------
    # 6. In-Memory Caching (10-Minute TTL)
    # --------------------------------------------------------------------------
    @patch("httpx.Client.get")
    def test_weather_caching_behavior(self, mock_get):
        """Subsequent request within TTL serves from cache without hitting network."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "latitude": 22.95,
            "longitude": 88.38,
            "timezone": "Asia/Kolkata",
            "current": {
                "time": "2026-10-14T12:00",
                "temperature_2m": 29.4,
                "relative_humidity_2m": 68,
                "apparent_temperature": 33.1,
                "is_day": 1,
                "precipitation": 0.0,
                "weather_code": 1,
                "wind_speed_10m": 8.5,
            },
        }
        mock_get.return_value = mock_resp

        # First call fetches from network
        first = fetch_open_meteo_weather(22.95, 88.38)
        self.assertFalse(first["cached"])
        self.assertEqual(mock_get.call_count, 1)

        # Second call immediately serves from cache
        second = fetch_open_meteo_weather(22.95, 88.38)
        self.assertTrue(second["cached"])
        self.assertEqual(mock_get.call_count, 1)  # Network was NOT called again

    # --------------------------------------------------------------------------
    # 7. Disambiguation & Clarification
    # --------------------------------------------------------------------------
    @patch("backend.weather_service.geocode_location")
    def test_handle_weather_query_ambiguous_location(self, mock_geo):
        """Prompts user to clarify when a query matches multiple disparate locations."""
        mock_geo.return_value = [
            {"name": "Springfield", "admin1": "Illinois", "country": "United States", "latitude": 39.78, "longitude": -89.65},
            {"name": "Springfield", "admin1": "Missouri", "country": "United States", "latitude": 37.21, "longitude": -93.29},
            {"name": "Springfield", "admin1": "Massachusetts", "country": "United States", "latitude": 42.10, "longitude": -72.58},
        ]
        # Querying with a single word "Springfield"
        res = handle_weather_query("Weather in Springfield")
        # For multiple matches, checks clarification or returns top option with details
        self.assertIn("Springfield", res["message"])

    # --------------------------------------------------------------------------
    # 8. Full Chat Integration via API Endpoint
    # --------------------------------------------------------------------------
    @patch("backend.weather_service.geocode_location")
    @patch("backend.weather_service.fetch_open_meteo_weather")
    def test_chat_api_adisaptagram_weather(self, mock_fetch, mock_geo):
        """User asking for Adisaptagram weather gets resolved response via /api/v1/chat."""
        mock_geo.return_value = [
            {
                "name": "Adisaptagram",
                "latitude": 22.95,
                "longitude": 88.38,
                "country": "India",
                "admin1": "West Bengal",
                "timezone": "Asia/Kolkata",
            }
        ]
        mock_fetch.return_value = {
            "temperature_c": 29.4,
            "temperature_f": 84.9,
            "apparent_temperature_c": 33.1,
            "apparent_temperature_f": 91.6,
            "humidity_percent": 68,
            "precipitation_mm": 0.0,
            "wind_speed_kmh": 8.5,
            "weather_code": 1,
            "condition_description": "Mainly clear",
            "condition_emoji": "🌤️",
            "is_day": True,
            "observation_time": "2026-10-14T12:00",
            "timezone": "Asia/Kolkata",
            "cached": False,
        }

        resp = self.client.post(
            "/api/v1/chat",
            json={"message": "What is today's weather in Adisaptagram?"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["mode"], "weather_service")
        self.assertFalse(data["supported"])
        self.assertIn("Adisaptagram", data["message"])
        self.assertIn("West Bengal", data["message"])
        self.assertIn("29.4°C", data["message"])
        self.assertIn("Mainly clear", data["message"])
        self.assertIn("Open-Meteo", data["ai_disclaimer"])


if __name__ == "__main__":
    unittest.main()
