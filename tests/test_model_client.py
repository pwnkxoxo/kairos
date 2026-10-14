"""
Unit tests for backend/model_client.py
Tests validation, parsing, error sanitization, and safety checks with zero external network calls.
"""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.model_client import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_TIMEOUT_MS,
    MAX_CONTENT_LENGTH,
    BaseModelProvider,
    GemmaPlan,
    GemmaProvider,
    ModelClientError,
    ModelRequest,
    ModelResponse,
    PLANNING_PROMPT_TEMPLATE,
    format_context_slice,
    generate_gemma_plan,
    get_active_provider,
    get_model_name,
    get_model_timeout_ms,
    parse_and_validate_model_output,
    sanitize_provider_error,
)


class TestModelClient(unittest.TestCase):

    def test_default_model_name(self):
        """Model name defaults to gemma-4-26b-a4b-it."""
        self.assertEqual(DEFAULT_MODEL, "gemma-4-26b-a4b-it")

    def test_parse_valid_supported_output(self):
        """Valid JSON for supported request is correctly parsed into GemmaPlan."""
        raw = """{
            "supported": true,
            "assistant_message": "I generated a project note for you.",
            "plan_description": "Create demo.txt with meeting agenda.",
            "file_content": "ECHO Meeting Agenda:\\n1. Review AI safety\\n2. Test approval gate"
        }"""
        plan = parse_and_validate_model_output(raw, model_name="gemma-4-26b-a4b-it")

        self.assertTrue(plan.supported)
        self.assertEqual(plan.assistant_message, "I generated a project note for you.")
        self.assertEqual(plan.plan_description, "Create demo.txt with meeting agenda.")
        self.assertIn("ECHO Meeting Agenda", plan.file_content)
        self.assertEqual(plan.mode, "gemma")
        self.assertEqual(plan.model_name, "gemma-4-26b-a4b-it")

    def test_parse_valid_output_with_markdown_fences(self):
        """Model output wrapped in markdown code fences is cleaned and parsed."""
        raw = """```json
        {
            "supported": true,
            "assistant_message": "Created notes.",
            "plan_description": "Write notes to demo.txt",
            "file_content": "Clean content inside fences"
        }
        ```"""
        plan = parse_and_validate_model_output(raw, model_name="gemma-4-26b-a4b-it")
        self.assertTrue(plan.supported)
        self.assertEqual(plan.file_content, "Clean content inside fences")
        self.assertEqual(plan.target_file, "demo.txt")

    def test_parse_valid_output_with_custom_target_file(self):
        """Model proposing custom relative target_file is preserved in GemmaPlan."""
        raw = """{
            "supported": true,
            "assistant_message": "Created study notes.",
            "plan_description": "Write study notes to notes/biology.txt",
            "file_content": "Chapter 1 Biology Notes",
            "target_file": "notes/biology.txt"
        }"""
        plan = parse_and_validate_model_output(raw, model_name="gemma-4-26b-a4b-it")
        self.assertTrue(plan.supported)
        self.assertEqual(plan.target_file, "notes/biology.txt")

    def test_reject_unsafe_model_proposed_target_file(self):
        """Model proposing path traversal or absolute target_file raises ModelClientError."""
        bad_outputs = [
            '{"supported": true, "assistant_message": "hacked", "file_content": "bad", "target_file": "../../etc/passwd"}',
            '{"supported": true, "assistant_message": "hacked", "file_content": "bad", "target_file": "/var/log/syslog"}',
        ]
        for raw in bad_outputs:
            with self.subTest(raw=raw):
                with self.assertRaises(ModelClientError):
                    parse_and_validate_model_output(raw, model_name="test")

    def test_parse_valid_unsupported_output(self):
        """Unsupported request sets supported=False and clears executable content."""
        raw = """{
            "supported": false,
            "assistant_message": "I only support writing text files.",
            "plan_description": "Should be ignored",
            "file_content": "Malicious code should be cleared"
        }"""
        plan = parse_and_validate_model_output(raw, model_name="gemma-4-26b-a4b-it")
        self.assertFalse(plan.supported)
        self.assertEqual(plan.assistant_message, "I only support writing text files.")
        self.assertEqual(plan.file_content, "")
        self.assertEqual(plan.plan_description, "")

    def test_reject_empty_response(self):
        """Empty or whitespace response raises ModelClientError."""
        with self.assertRaises(ModelClientError):
            parse_and_validate_model_output("", model_name="test")
        with self.assertRaises(ModelClientError):
            parse_and_validate_model_output("   \n  ", model_name="test")

    def test_reject_invalid_json(self):
        """Malformed JSON raises ModelClientError."""
        with self.assertRaises(ModelClientError):
            parse_and_validate_model_output("This is plain text, not JSON", model_name="test")

    def test_reject_non_dict_json(self):
        """JSON that is a list or primitive raises ModelClientError."""
        with self.assertRaises(ModelClientError):
            parse_and_validate_model_output('["supported", true]', model_name="test")

    def test_reject_missing_fields(self):
        """Missing required fields raises ModelClientError."""
        with self.assertRaises(ModelClientError):
            parse_and_validate_model_output('{"supported": true}', model_name="test")

    def test_reject_empty_file_content_when_supported(self):
        """Supported request with blank file_content raises ModelClientError."""
        raw = '{"supported": true, "assistant_message": "Ok", "file_content": "   "}'
        with self.assertRaises(ModelClientError):
            parse_and_validate_model_output(raw, model_name="test")

    def test_reject_oversized_content(self):
        """Content exceeding MAX_CONTENT_LENGTH raises ModelClientError."""
        raw = f'{{"supported": true, "assistant_message": "Ok", "file_content": "{"x" * (MAX_CONTENT_LENGTH + 10)}"}}'
        with self.assertRaises(ModelClientError):
            parse_and_validate_model_output(raw, model_name="test")

    def test_sanitize_provider_error_auth(self):
        """Auth errors produce clear messages without leaking secrets."""
        err = Exception("401 Client Error: Invalid API key secret_123456789")
        sanitized = sanitize_provider_error(err)
        self.assertIn("Authentication failed", sanitized)
        self.assertNotIn("secret_123456789", sanitized)

    def test_sanitize_provider_error_quota(self):
        """Quota errors produce friendly rate limit message."""
        err = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded")
        sanitized = sanitize_provider_error(err)
        self.assertIn("rate limit or quota exceeded", sanitized)

    def test_sanitize_provider_error_timeout(self):
        """Timeout errors produce clear timeout message."""
        err = Exception("Connection timed out after 30 seconds")
        sanitized = sanitize_provider_error(err)
        self.assertIn("timed out", sanitized)

    def test_sanitize_provider_error_internal(self):
        """Internal 500 errors produce friendly guidance without exposing provider traces."""
        err = Exception("500 INTERNAL - Internal error encountered")
        sanitized = sanitize_provider_error(err)
        self.assertIn("internal error", sanitized.lower())

    def test_default_model_timeout_ms(self):
        """Request timeout defaults to 60,000 ms."""
        self.assertEqual(DEFAULT_MODEL_TIMEOUT_MS, 60000)
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {}, clear=False):
            if "GEMINI_REQUEST_TIMEOUT_MS" in os.environ:
                del os.environ["GEMINI_REQUEST_TIMEOUT_MS"]
            self.assertEqual(get_model_timeout_ms(), 60000)

    def test_custom_model_timeout_ms(self):
        """Request timeout can be customized via GEMINI_REQUEST_TIMEOUT_MS."""
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"GEMINI_REQUEST_TIMEOUT_MS": "45000"}):
            self.assertEqual(get_model_timeout_ms(), 45000)
        with patch.dict(os.environ, {"GEMINI_REQUEST_TIMEOUT_MS": "invalid"}):
            self.assertEqual(get_model_timeout_ms(), 60000)

    def test_pdf_prompt_template_strict_grounding_rules(self):
        """PDF analysis prompt strictly prohibits guessing and requires citation."""
        from backend.model_client import PDF_ANALYSIS_PROMPT_TEMPLATE
        self.assertIn("STRICT GROUNDING IN EXTRACTED TEXT", PDF_ANALYSIS_PROMPT_TEMPLATE)
        self.assertIn("Do NOT invent missing equations", PDF_ANALYSIS_PROMPT_TEMPLATE)
        self.assertIn("HANDLING AMBIGUOUS, GARBLED, OR INSUFFICIENT EVIDENCE", PDF_ANALYSIS_PROMPT_TEMPLATE)
        self.assertIn("NEVER claim that a mathematical proof", PDF_ANALYSIS_PROMPT_TEMPLATE)

    def test_planning_prompt_quality_guidelines(self):
        """Planning prompt contains explicit instructions for directness and uncertainty handling."""
        self.assertIn("RESPONSE QUALITY GUIDELINES", PLANNING_PROMPT_TEMPLATE)
        self.assertIn("DIRECTNESS & CONCISENESS", PLANNING_PROMPT_TEMPLATE)
        self.assertIn("UNCERTAINTY & AMBIGUITY", PLANNING_PROMPT_TEMPLATE)
        self.assertIn("AVOID REPETITION", PLANNING_PROMPT_TEMPLATE)

    def test_format_context_slice_bounding(self):
        """Context slice retains head and tail of oversized turns with an omission marker."""
        slice_data = [
            {"role": "user", "content": "Short question"},
            {"role": "assistant", "content": "START_PREFIX_" + ("X" * 1500) + "_END_SUFFIX"},
        ]
        formatted = format_context_slice(slice_data, max_turn_chars=200)
        self.assertIn("User: Short question", formatted)
        self.assertIn("START_PREFIX_", formatted)
        self.assertIn("_END_SUFFIX", formatted)
        self.assertIn("... [middle content omitted for brevity] ...", formatted)

    def test_format_context_slice_preserves_end_of_long_turn(self):
        """Verifies that important constraints or decisions at the end of a long message are retained."""
        long_content = (
            "We discussed multiple architectural considerations earlier in the turn. "
            * 25  # ~1800 chars
            + "\nCRITICAL_REQUIREMENT: Ensure database is SQLite only and file name is notes.txt"
        )
        slice_data = [
            {"role": "user", "content": long_content},
        ]
        formatted = format_context_slice(slice_data, max_turn_chars=500)
        self.assertIn("CRITICAL_REQUIREMENT: Ensure database is SQLite only and file name is notes.txt", formatted)

    def test_format_context_slice_order_and_chronology(self):
        """Context slice preserves chronological order of conversation messages."""
        slice_data = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"},
        ]
        formatted = format_context_slice(slice_data)
        first_idx = formatted.find("First message")
        second_idx = formatted.find("Second message")
        third_idx = formatted.find("Third message")
        self.assertTrue(first_idx < second_idx < third_idx)

    def test_format_context_slice_empty(self):
        """Empty or None context slice returns empty string."""
        self.assertEqual(format_context_slice(None), "")
        self.assertEqual(format_context_slice([]), "")

    def test_base_provider_abstract_methods(self):
        """BaseModelProvider cannot be used without concrete implementation."""
        class DummyProvider(BaseModelProvider):
            pass

        p = DummyProvider()
        with self.assertRaises(NotImplementedError):
            p.is_available()
        with self.assertRaises(NotImplementedError):
            p.get_model_name()
        with self.assertRaises(NotImplementedError):
            p.generate(ModelRequest(prompt="test"))

    def test_gemma_provider_generate_with_mock_client(self):
        """GemmaProvider generates valid ModelResponse using injected mock client."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"supported": false, "assistant_message": "Hello there"}'
        mock_client.models.generate_content.return_value = mock_resp

        provider = GemmaProvider(client=mock_client)
        req = ModelRequest(prompt="Say hi")
        resp = provider.generate(req)

        self.assertEqual(resp.provider_name, "gemma")
        self.assertIn("Hello there", resp.text)
        mock_client.models.generate_content.assert_called_once()

    def test_generate_gemma_plan_with_provider_injection(self):
        """generate_gemma_plan works seamlessly when passed a custom provider."""
        from unittest.mock import MagicMock
        mock_provider = MagicMock(spec=BaseModelProvider)
        mock_provider.is_available.return_value = True
        mock_provider.get_model_name.return_value = "gemma-4-26b-a4b-it"
        mock_provider.generate.return_value = ModelResponse(
            text='{"supported": true, "assistant_message": "Done", "plan_description": "Create notes", "file_content": "Note content"}',
            model_name="gemma-4-26b-a4b-it",
            provider_name="gemma",
        )

        plan = generate_gemma_plan("Write note", provider=mock_provider)
        self.assertTrue(plan.supported)
        self.assertEqual(plan.assistant_message, "Done")
        self.assertEqual(plan.file_content, "Note content")

    def test_generate_gemma_plan_preserves_full_user_request(self):
        """Verifies the latest user prompt is never truncated when injected into the planning prompt."""
        from unittest.mock import MagicMock
        mock_provider = MagicMock(spec=BaseModelProvider)
        mock_provider.is_available.return_value = True
        mock_provider.get_model_name.return_value = "gemma-4-26b-a4b-it"

        captured_request = None
        def mock_generate(req):
            nonlocal captured_request
            captured_request = req
            return ModelResponse(
                text='{"supported": false, "assistant_message": "Understood"}',
                model_name="gemma-4-26b-a4b-it",
            )
        mock_provider.generate.side_effect = mock_generate

        very_long_user_prompt = "Create a detailed implementation: " + ("Requirement detail. " * 50)
        generate_gemma_plan(very_long_user_prompt, provider=mock_provider)

        self.assertIsNotNone(captured_request)
        self.assertIn(very_long_user_prompt.strip(), captured_request.prompt)

    def test_gemma_provider_handles_timeout(self):
        """GemmaProvider cleanly sanitizes and raises ModelClientError on timeout."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("504 Gateway Timeout: Deadline exceeded after 60000ms")

        provider = GemmaProvider(client=mock_client)
        req = ModelRequest(prompt="Say hi")

        with self.assertRaises(ModelClientError) as ctx:
            provider.generate(req)
        self.assertIn("timed out", str(ctx.exception).lower())

    def test_gemma_provider_handles_rate_limit(self):
        """GemmaProvider cleanly sanitizes and raises ModelClientError on 429 quota exhaustion."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

        provider = GemmaProvider(client=mock_client)
        req = ModelRequest(prompt="Say hi")

        with self.assertRaises(ModelClientError) as ctx:
            provider.generate(req)
        self.assertIn("rate limit or quota exceeded", str(ctx.exception).lower())

    def test_generate_gemma_plan_handles_malformed_provider_json(self):
        """generate_gemma_plan raises ModelClientError with clean message when provider returns invalid JSON."""
        from unittest.mock import MagicMock
        mock_provider = MagicMock(spec=BaseModelProvider)
        mock_provider.is_available.return_value = True
        mock_provider.get_model_name.return_value = "gemma-4-26b-a4b-it"
        mock_provider.generate.return_value = ModelResponse(
            text="I am not returning JSON, just plain conversation text.",
            model_name="gemma-4-26b-a4b-it",
        )

        with self.assertRaises(ModelClientError) as ctx:
            generate_gemma_plan("User message", provider=mock_provider)
        self.assertIn("invalid JSON", str(ctx.exception))

    def test_parse_latex_math_formulas_repaired_successfully(self):
        """LaTeX formulas with \frac, \begin, \theta are safely repaired and preserved."""
        raw = r'''{
            "supported": false,
            "intent": "general_question",
            "assistant_message": "The formula is $\frac{a}{b}$ and $\begin{matrix} 1 & 0 \\ 0 & 1 \end{matrix}$ with angle $\theta$."
        }'''
        plan = parse_and_validate_model_output(raw, model_name="gemma-4-26b-a4b-it")
        self.assertFalse(plan.supported)
        self.assertIn("frac", plan.assistant_message)
        self.assertIn("theta", plan.assistant_message)

    def test_parse_long_assistant_message(self):
        """Long assistant messages up to 64000 characters parse without schema validation error."""
        long_text = "Here is a detailed explanation of the concept:\n" + ("Point explanation. " * 1500)
        raw = f'''{{
            "supported": false,
            "intent": "general_question",
            "assistant_message": "{long_text}"
        }}'''
        plan = parse_and_validate_model_output(raw, model_name="gemma-4-26b-a4b-it")
        self.assertFalse(plan.supported)
        self.assertGreater(len(plan.assistant_message), 10000)

    def test_parse_json_with_unescaped_quotes_recovery(self):
        """JSON output with unescaped internal double quotes recovers assistant_message."""
        raw = '''{
            "supported": false,
            "intent": "general_question",
            "assistant_message": "He said "hello world" to everyone.",
            "plan_description": ""
        }'''
        plan = parse_and_validate_model_output(raw, model_name="gemma-4-26b-a4b-it")
        self.assertFalse(plan.supported)
        self.assertIn("hello world", plan.assistant_message)


if __name__ == "__main__":
    unittest.main()
