"""
Unit tests for ECHO FastAPI Backend with Gemma 4 AI & Demo Mode (Phase 3)
Tests all API endpoints, permission controls, and safety guarantees:
  1. Health check returns 200 and valid payload with mode and model info.
  2. Demo mode enabled when API key is missing, clearly labelled.
  3. Gemma mode: supported requests produce pending plans without touching disk.
  4. Gemma mode: unsupported requests do NOT produce executable pending actions.
  5. Approval writes the validated model content to disk.
  6. Rejection marks rejected and creates no file.
  7. Provider errors return HTTP 502 with clear error details instead of false success.
  8. Invalid action ID returns 404.
  9. Duplicate approval/rejection is prevented with 400.
  10. Request validation: empty message returns 422.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.agent import WORKSPACE_DIR
from backend.api import app, clear_pending_actions
from backend.model_client import GemmaPlan, ModelClientError


class TestEchoAPI(unittest.TestCase):

    def setUp(self):
        """Runs BEFORE each test: initialize test client, clean memory and disk."""
        self.client = TestClient(app)
        self.target_file = WORKSPACE_DIR / "demo.txt"

        clear_pending_actions()
        if self.target_file.exists():
            self.target_file.unlink()

        self.addCleanup(self.cleanup_resources)

    def cleanup_resources(self):
        """Cleanup helper: remove any files and reset action store."""
        if self.target_file.exists():
            self.target_file.unlink()
        clear_pending_actions()

    # --------------------------------------------------------------------------
    # 1. Health check
    # --------------------------------------------------------------------------
    def test_health_check(self):
        """GET /health should return 200 OK with service details and mode."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("service"), "KAIROS")
        self.assertEqual(data.get("version"), "0.5.0")
        self.assertIn("mode", data)
        self.assertIn("model", data)

    # --------------------------------------------------------------------------
    # 2. Demo mode (when API key is not configured)
    # --------------------------------------------------------------------------
    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_chat_demo_mode_when_unconfigured(self, _mock_is_cfg):
        """When GEMINI_API_KEY is not configured, safe demo mode is used and labelled."""
        response = self.client.post("/api/v1/chat", json={"message": "Please write a test note."})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIsNotNone(data.get("action_id"))
        self.assertEqual(data["status"], "pending_approval")
        self.assertEqual(data["mode"], "demo")
        self.assertIn("Demo mode", data["ai_disclaimer"])
        self.assertTrue(len(data["plan"]) > 0)
        self.assertEqual(data["plan"][0]["target_file"], "demo.txt")

        # Crucial security check: file must NOT exist before approval
        self.assertFalse(
            self.target_file.exists(),
            "workspace/demo.txt must NOT be created when only generating a plan!"
        )

    # --------------------------------------------------------------------------
    # 3. Gemma mode: Supported request creates pending plan
    # --------------------------------------------------------------------------
    @patch("backend.api.is_gemini_configured", return_value=True)
    @patch("backend.api.generate_gemma_plan")
    def test_chat_gemma_supported_request(self, mock_gen, _mock_is_cfg):
        """Supported request via Gemma creates a pending action with model content."""
        mock_gen.return_value = GemmaPlan(
            supported=True,
            assistant_message="I have prepared your meeting notes.",
            plan_description="Create meeting notes in demo.txt",
            file_content="Agenda:\n1. Sprint planning\n2. AI permission verification",
            mode="gemma",
            model_name="gemma-4-26b-a4b-it",
        )

        response = self.client.post("/api/v1/chat", json={"message": "Create meeting notes"})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIsNotNone(data["action_id"])
        self.assertEqual(data["status"], "pending_approval")
        self.assertEqual(data["mode"], "gemma")
        self.assertTrue(data["supported"])
        self.assertIn("Gemma 4", data["ai_disclaimer"])
        self.assertEqual(data["file_preview"], "Agenda:\n1. Sprint planning\n2. AI permission verification")

        # Crucial: NO file written on disk during planning
        self.assertFalse(self.target_file.exists())

    # --------------------------------------------------------------------------
    # 4. Gemma mode: Unsupported request does NOT create pending action
    # --------------------------------------------------------------------------
    @patch("backend.api.is_gemini_configured", return_value=True)
    @patch("backend.api.generate_gemma_plan")
    def test_chat_gemma_unsupported_request_no_action(self, mock_gen, _mock_is_cfg):
        """Unsupported request returns status 'unsupported' and NO action_id."""
        mock_gen.return_value = GemmaPlan(
            supported=False,
            assistant_message="I only support creating text files in the workspace.",
            plan_description="",
            file_content="",
            mode="gemma",
            model_name="gemma-4-26b-a4b-it",
        )

        response = self.client.post("/api/v1/chat", json={"message": "What is the capital of France?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIsNone(data.get("action_id"))
        self.assertEqual(data["status"], "unsupported")
        self.assertFalse(data["supported"])
        self.assertEqual(len(data["plan"]), 0)
        self.assertFalse(self.target_file.exists())

    # --------------------------------------------------------------------------
    # 5. Gemma mode: Provider errors return HTTP 502
    # --------------------------------------------------------------------------
    @patch("backend.api.is_gemini_configured", return_value=True)
    @patch("backend.api.generate_gemma_plan")
    def test_chat_gemma_provider_error_handled(self, mock_gen, _mock_is_cfg):
        """Provider failure returns 502 Bad Gateway and does not falsely claim success."""
        mock_gen.side_effect = ModelClientError("Google AI rate limit or quota exceeded.")

        response = self.client.post("/api/v1/chat", json={"message": "Create a file"})
        self.assertEqual(response.status_code, 502)
        data = response.json()
        self.assertIn("Google AI rate limit", data["detail"])

    # --------------------------------------------------------------------------
    # 6. Approval workflow: writes model-generated content
    # --------------------------------------------------------------------------
    @patch("backend.api.is_gemini_configured", return_value=True)
    @patch("backend.api.generate_gemma_plan")
    def test_approval_writes_validated_gemma_content(self, mock_gen, _mock_is_cfg):
        """Approving an action writes the exact model content and verifies disk."""
        mock_gen.return_value = GemmaPlan(
            supported=True,
            assistant_message="Draft ready.",
            plan_description="Write checklist",
            file_content="[ ] Security audit\n[ ] Permissions check",
            mode="gemma",
            model_name="gemma-4-26b-a4b-it",
        )

        # 1. Create plan
        chat_resp = self.client.post("/api/v1/chat", json={"message": "Checklist file"})
        action_id = chat_resp.json()["action_id"]

        # 2. Approve
        approve_resp = self.client.post(f"/api/v1/actions/{action_id}/approve")
        self.assertEqual(approve_resp.status_code, 200)
        data = approve_resp.json()

        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["action_id"], action_id)

        # 3. Disk verification
        self.assertTrue(self.target_file.exists())
        content = self.target_file.read_text(encoding="utf-8")
        self.assertEqual(content, "[ ] Security audit\n[ ] Permissions check")

    # --------------------------------------------------------------------------
    # 7. Rejection workflow: guarantees no files written
    # --------------------------------------------------------------------------
    @patch("backend.api.is_gemini_configured", return_value=True)
    @patch("backend.api.generate_gemma_plan")
    def test_rejection_marks_rejected_and_creates_no_file(self, mock_gen, _mock_is_cfg):
        """Rejecting an action marks status rejected with zero files touched."""
        mock_gen.return_value = GemmaPlan(
            supported=True,
            assistant_message="Draft ready.",
            plan_description="Write file",
            file_content="Will not be written",
            mode="gemma",
            model_name="gemma-4-26b-a4b-it",
        )

        chat_resp = self.client.post("/api/v1/chat", json={"message": "Create file"})
        action_id = chat_resp.json()["action_id"]

        reject_resp = self.client.post(f"/api/v1/actions/{action_id}/reject")
        self.assertEqual(reject_resp.status_code, 200)
        data = reject_resp.json()

        self.assertEqual(data["status"], "rejected")
        self.assertFalse(self.target_file.exists())

    # --------------------------------------------------------------------------
    # 8. Invalid action ID handling
    # --------------------------------------------------------------------------
    def test_unknown_action_id_returns_404(self):
        """Approving or rejecting a non-existent action ID should return 404."""
        approve_resp = self.client.post("/api/v1/actions/unknown-uuid-12345/approve")
        self.assertEqual(approve_resp.status_code, 404)

        reject_resp = self.client.post("/api/v1/actions/unknown-uuid-12345/reject")
        self.assertEqual(reject_resp.status_code, 404)

    # --------------------------------------------------------------------------
    # 9. Duplicate processing prevention
    # --------------------------------------------------------------------------
    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_cannot_approve_or_reject_already_processed_action(self, _mock_is_cfg):
        """Actions that are already completed or rejected cannot be processed again."""
        chat_resp = self.client.post("/api/v1/chat", json={"message": "Create file."})
        action_id = chat_resp.json()["action_id"]

        self.client.post(f"/api/v1/actions/{action_id}/approve")

        # Second approval should fail with 400 Bad Request
        dup_approve = self.client.post(f"/api/v1/actions/{action_id}/approve")
        self.assertEqual(dup_approve.status_code, 400)

        # Rejecting completed action should fail with 400
        dup_reject = self.client.post(f"/api/v1/actions/{action_id}/reject")
        self.assertEqual(dup_reject.status_code, 400)

    # --------------------------------------------------------------------------
    # 11. Regression: General question persistence & conversation_id
    # --------------------------------------------------------------------------
    @patch("backend.api.generate_gemma_plan")
    @patch("backend.api.is_gemini_configured", return_value=True)
    def test_chat_general_question_persists_messages_and_returns_conversation_id(
        self, _mock_is_cfg, mock_gen_plan
    ):
        """General questions (e.g. 'hello') return 200 with supported=False and persist both turns in SQLite."""
        mock_gen_plan.return_value = GemmaPlan(
            supported=False,
            assistant_message="Hello! I am ECHO, your permission-gated AI assistant.",
            plan_description="",
            file_content="",
            intent="general_question",
            mode="gemma",
            model_name="gemma-4-26b-a4b-it",
        )

        resp = self.client.post("/api/v1/chat", json={"message": "hello"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["supported"])
        self.assertIn("Hello! I am ECHO", data["message"])
        self.assertIsNotNone(data.get("conversation_id"))

        conv_id = data["conversation_id"]
        # Verify both messages are in the conversation detail
        conv_resp = self.client.get(f"/api/v1/conversations/{conv_id}")
        self.assertEqual(conv_resp.status_code, 200)
        conv_data = conv_resp.json()
        self.assertEqual(len(conv_data["messages"]), 2)
        self.assertEqual(conv_data["messages"][0]["role"], "user")
        self.assertEqual(conv_data["messages"][0]["content"], "hello")
        self.assertEqual(conv_data["messages"][1]["role"], "assistant")
        self.assertIn("Hello! I am ECHO", conv_data["messages"][1]["content"])

    # --------------------------------------------------------------------------
    # 12. Regression: File creation plan persistence & pending status
    # --------------------------------------------------------------------------
    @patch("backend.api.generate_gemma_plan")
    @patch("backend.api.is_gemini_configured", return_value=True)
    def test_chat_file_creation_persists_messages_and_pending_plan(
        self, _mock_is_cfg, mock_gen_plan
    ):
        """File creation request returns 200 with pending_approval, plan steps, and persists turns."""
        mock_gen_plan.return_value = GemmaPlan(
            supported=True,
            assistant_message="I prepared a welcome note for you.",
            plan_description="Create welcome note in demo.txt",
            file_content="Welcome to Project ECHO!",
            intent="create_file",
            mode="gemma",
            model_name="gemma-4-26b-a4b-it",
        )

        resp = self.client.post("/api/v1/chat", json={"message": "create a welcome note"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["supported"])
        self.assertEqual(data["status"], "pending_approval")
        self.assertIsNotNone(data.get("action_id"))
        self.assertIsNotNone(data.get("conversation_id"))

        conv_id = data["conversation_id"]
        conv_resp = self.client.get(f"/api/v1/conversations/{conv_id}")
        self.assertEqual(conv_resp.status_code, 200)
        conv_data = conv_resp.json()
        self.assertEqual(len(conv_data["messages"]), 2)
        self.assertEqual(conv_data["messages"][1]["role"], "assistant")

    # --------------------------------------------------------------------------
    # 13. Regression: Model provider failure returns 502 with X-Conversation-Id
    # --------------------------------------------------------------------------
    @patch("backend.api.generate_gemma_plan")
    @patch("backend.api.is_gemini_configured", return_value=True)
    def test_chat_provider_failure_returns_502_with_conversation_header(
        self, _mock_is_cfg, mock_gen_plan
    ):
        """When Gemma raises ModelClientError, endpoint returns 502 with X-Conversation-Id and saves user message."""
        mock_gen_plan.side_effect = ModelClientError("Unable to reach Google AI service due to network failure.")

        resp = self.client.post("/api/v1/chat", json={"message": "create another welcome note"})
        self.assertEqual(resp.status_code, 502)
        self.assertIn("network failure", resp.json()["detail"])
        conv_id = resp.headers.get("X-Conversation-Id")
        self.assertIsNotNone(conv_id)

        # Check that user prompt was safely preserved in the database
        conv_resp = self.client.get(f"/api/v1/conversations/{conv_id}")
        self.assertEqual(conv_resp.status_code, 200)
        conv_data = conv_resp.json()
        self.assertEqual(len(conv_data["messages"]), 1)
        self.assertEqual(conv_data["messages"][0]["content"], "create another welcome note")

    # --------------------------------------------------------------------------
    # 14. Regression: Conversation continuity across multiple turns
    # --------------------------------------------------------------------------
    @patch("backend.api.generate_gemma_plan")
    @patch("backend.api.is_gemini_configured", return_value=True)
    def test_chat_conversation_continuity(self, _mock_is_cfg, mock_gen_plan):
        """Sequential turns using the same conversation_id accumulate in the same conversation."""
        mock_gen_plan.return_value = GemmaPlan(
            supported=False,
            assistant_message="Answer 1",
            plan_description="",
            file_content="",
            intent="general_question",
            mode="gemma",
            model_name="gemma-4-26b-a4b-it",
        )

        resp1 = self.client.post("/api/v1/chat", json={"message": "Turn 1"})
        conv_id = resp1.json()["conversation_id"]

        mock_gen_plan.return_value = GemmaPlan(
            supported=False,
            assistant_message="Answer 2",
            plan_description="",
            file_content="",
            intent="general_question",
            mode="gemma",
            model_name="gemma-4-26b-a4b-it",
        )

        resp2 = self.client.post("/api/v1/chat", json={"message": "Turn 2", "conversation_id": conv_id})
        self.assertEqual(resp2.json()["conversation_id"], conv_id)

        conv_resp = self.client.get(f"/api/v1/conversations/{conv_id}")
        conv_data = conv_resp.json()
        self.assertEqual(len(conv_data["messages"]), 4)

    # --------------------------------------------------------------------------
    # 15. Plan 5A Regression: Action persistence across fresh application instances
    # --------------------------------------------------------------------------
    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_action_persists_across_fresh_client_instance(self, _mock_is_cfg):
        """Action created by one client instance can be approved by a fresh new client instance (simulating restart)."""
        # Client 1 creates plan
        chat_resp = self.client.post("/api/v1/chat", json={"message": "Save this file later."})
        action_id = chat_resp.json()["action_id"]

        # Simulate backend reboot / separate process with fresh TestClient
        new_client = TestClient(app)

        # Inspect status from fresh client
        status_resp = new_client.get(f"/api/v1/actions/{action_id}")
        self.assertEqual(status_resp.status_code, 200)
        self.assertEqual(status_resp.json()["status"], "pending_approval")

        # Approve using fresh client
        approve_resp = new_client.post(f"/api/v1/actions/{action_id}/approve")
        self.assertEqual(approve_resp.status_code, 200)
        self.assertEqual(approve_resp.json()["status"], "completed")

        # Verify disk
        self.assertTrue(self.target_file.exists())

    # --------------------------------------------------------------------------
    # 16. Plan 5A Regression: Expired actions cannot execute
    # --------------------------------------------------------------------------
    def test_expired_action_cannot_be_approved_or_executed(self):
        """Actions past their 24h expiration timestamp fail approval without touching filesystem."""
        from backend.storage import store_action
        conv_id = "test-exp-conv"
        action_id = "act-expired-test"
        store_action(
            action_id=action_id,
            conversation_id=conv_id,
            user_message="Expired work",
            steps=[{"step_number": 1, "action": "create_file", "target_file": "demo.txt", "content": "stale"}],
            mode="demo",
            ttl_seconds=-1,  # Expired immediately
        )

        approve_resp = self.client.post(f"/api/v1/actions/{action_id}/approve")
        self.assertEqual(approve_resp.status_code, 400)
        self.assertIn("expired", approve_resp.json()["detail"].lower())
        self.assertFalse(self.target_file.exists())

        status_resp = self.client.get(f"/api/v1/actions/{action_id}")
        self.assertEqual(status_resp.status_code, 200)
        self.assertEqual(status_resp.json()["status"], "expired")

    # --------------------------------------------------------------------------
    # 17. Plan 5A Regression: Concurrent or duplicate approval requests execute at most once
    # --------------------------------------------------------------------------
    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_concurrent_or_duplicate_approvals_execute_once(self, _mock_is_cfg):
        """Simultaneous or duplicate approvals execute at most once, second call receives 400."""
        chat_resp = self.client.post("/api/v1/chat", json={"message": "Write demo"})
        action_id = chat_resp.json()["action_id"]

        r1 = self.client.post(f"/api/v1/actions/{action_id}/approve")
        r2 = self.client.post(f"/api/v1/actions/{action_id}/approve")

        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 400)
        self.assertIn("already", r2.json()["detail"].lower())

    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_true_concurrent_approval_race(self, _mock_is_cfg):
        """
        Genuinely concurrent race condition test with 10 parallel threads.
        All threads synchronize on a threading.Barrier and hit /approve simultaneously.
        Guarantees:
          - Exactly 1 thread receives HTTP 200.
          - 9 threads receive HTTP 400.
          - File is written cleanly.
          - Database record transitions cleanly to 'completed'.
        """
        import concurrent.futures
        import threading

        chat_resp = self.client.post("/api/v1/chat", json={"message": "Concurrent demo"})
        action_id = chat_resp.json()["action_id"]

        num_threads = 10
        barrier = threading.Barrier(num_threads)
        results = []

        def worker():
            barrier.wait()
            # Send approval request
            resp = self.client.post(f"/api/v1/actions/{action_id}/approve")
            return resp.status_code, resp.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        status_codes = [r[0] for r in results]
        self.assertEqual(status_codes.count(200), 1, f"Expected exactly 1 HTTP 200, got: {status_codes}")
        self.assertEqual(status_codes.count(400), num_threads - 1, f"Expected {num_threads - 1} HTTP 400s, got: {status_codes}")

        # Check action in DB
        from backend.storage import get_action
        action = get_action(action_id)
        self.assertEqual(action["status"], "completed")
        self.assertTrue(self.target_file.exists())

    def test_startup_recovers_stranded_executing_actions_to_failed(self):
        """
        Simulate an app restart with a stranded 'executing' action in the database.
        Test that FastAPI startup handler recovers it to 'failed' requiring manual approval retry.
        """
        from backend.storage import store_action, claim_action_for_execution, get_action
        action_id = "act-crash-sim"
        store_action(
            action_id=action_id,
            conversation_id="conv-crash-test",
            user_message="Crash action",
            steps=[{"step_number": 1, "action": "create_file", "target_file": "demo.txt", "content": "hello"}],
            mode="demo",
        )
        claim_action_for_execution(action_id)
        self.assertEqual(get_action(action_id)["status"], "executing")

        # Trigger startup event handler
        from backend.api import on_startup
        on_startup()

        recovered_act = get_action(action_id)
        self.assertEqual(recovered_act["status"], "failed")
        self.assertIn("Manual approval retry required", recovered_act["error_message"])

        # Manual retry approval succeeds
        approve_resp = self.client.post(f"/api/v1/actions/{action_id}/approve")
        self.assertEqual(approve_resp.status_code, 200)
        self.assertEqual(approve_resp.json()["status"], "completed")
        self.assertTrue(self.target_file.exists())


    # --------------------------------------------------------------------------
    # 18. Plan 5A Regression: Path traversal in action step remains strictly blocked
    # --------------------------------------------------------------------------
    def test_path_traversal_in_action_step_is_blocked_and_marks_failed(self):
        """Action step attempting to write outside workspace raises PermissionError and marks action failed."""
        from backend.storage import store_action, get_action
        action_id = "act-traversal-hack"
        store_action(
            action_id=action_id,
            conversation_id="c-hack",
            user_message="Hack root",
            steps=[{"step_number": 1, "action": "create_file", "target_file": "../../escaped.txt", "content": "bad"}],
            mode="demo",
        )

        approve_resp = self.client.post(f"/api/v1/actions/{action_id}/approve")
        self.assertEqual(approve_resp.status_code, 400)
        self.assertIn("Security Alert", approve_resp.json()["detail"])

        act = get_action(action_id)
        self.assertEqual(act["status"], "failed")
        self.assertIn("Security Alert", act["error_message"])

    # --------------------------------------------------------------------------
    # 19. Stage 2 Tests: Generalized Safe File Targets
    # --------------------------------------------------------------------------
    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_stage2_demo_mode_requested_filename(self, _mock_is_cfg):
        """Demo mode extracts requested filename into action plan and executes to that target."""
        target_path = WORKSPACE_DIR / "notes/report.txt"
        if target_path.exists():
            target_path.unlink()

        try:
            chat_resp = self.client.post("/api/v1/chat", json={"message": "Write a note to 'notes/report.txt'"})
            self.assertEqual(chat_resp.status_code, 200)
            data = chat_resp.json()
            self.assertEqual(data["plan"][0]["target_file"], "notes/report.txt")
            action_id = data["action_id"]

            # Approve and check disk
            approve_resp = self.client.post(f"/api/v1/actions/{action_id}/approve")
            self.assertEqual(approve_resp.status_code, 200)
            res = approve_resp.json()["results"][0]
            self.assertEqual(res["target_file"], "workspace/notes/report.txt")

            self.assertTrue(target_path.exists())
            self.assertIn("Hello from KAIROS Web API!", target_path.read_text(encoding="utf-8"))
        finally:
            if target_path.exists():
                target_path.unlink()
            if target_path.parent.exists():
                import shutil
                shutil.rmtree(target_path.parent)

    @patch("backend.api.is_gemini_configured", return_value=True)
    @patch("backend.api.generate_gemma_plan")
    def test_stage2_gemma_mode_custom_target_execution_and_rejection(self, mock_gen, _mock_is_cfg):
        """Gemma mode supports custom target; approval executes exact target, rejection creates zero files."""
        target_path = WORKSPACE_DIR / "docs/overview.md"
        if target_path.exists():
            target_path.unlink()

        try:
            mock_gen.return_value = GemmaPlan(
                supported=True,
                assistant_message="Drafted custom file.",
                plan_description="Create nested markdown file",
                file_content="# Project Overview\nContent here.",
                target_file="docs/overview.md",
            )

            chat_resp = self.client.post("/api/v1/chat", json={"message": "Create docs/overview.md"})
            data = chat_resp.json()
            self.assertEqual(data["plan"][0]["target_file"], "docs/overview.md")
            action_id = data["action_id"]

            # 1. Test rejection creates no files
            reject_resp = self.client.post(f"/api/v1/actions/{action_id}/reject")
            self.assertEqual(reject_resp.status_code, 200)
            self.assertFalse(target_path.exists())

            # 2. Test approval on a fresh plan writes the exact target
            chat_resp2 = self.client.post("/api/v1/chat", json={"message": "Create docs/overview.md"})
            action_id2 = chat_resp2.json()["action_id"]

            approve_resp = self.client.post(f"/api/v1/actions/{action_id2}/approve")
            self.assertEqual(approve_resp.status_code, 200)
            self.assertTrue(target_path.exists())
            self.assertEqual(target_path.read_text(encoding="utf-8"), "# Project Overview\nContent here.")
        finally:
            if target_path.exists():
                target_path.unlink()
            if target_path.parent.exists():
                import shutil
                shutil.rmtree(target_path.parent)

    # --------------------------------------------------------------------------
    # Overwrite Authorization & Confirmation Security Tests
    # --------------------------------------------------------------------------
    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_negated_consent_does_not_create_overwrite_action(self, _mock_is_cfg):
        """Negated request ('Do not overwrite') does NOT create an overwrite plan or authorize replacement."""
        existing_target = WORKSPACE_DIR / "existing_doc.txt"
        existing_target.write_text("Existing critical data", encoding="utf-8")

        try:
            chat_resp = self.client.post("/api/v1/chat", json={"message": "Do not overwrite my file 'existing_doc.txt'"})
            self.assertEqual(chat_resp.status_code, 200)
            data = chat_resp.json()

            self.assertIsNone(data.get("action_id"))
            self.assertEqual(data["status"], "confirmation_required")
            self.assertFalse(data["supported"])
            self.assertEqual(data["plan"], [])
            self.assertIsNotNone(data.get("confirmation_token"))
            self.assertEqual(data.get("confirmation_target_file"), "existing_doc.txt")
            self.assertIn("already exists", data["message"])

            # Verify original file content was untouched
            self.assertEqual(existing_target.read_text(encoding="utf-8"), "Existing critical data")
        finally:
            if existing_target.exists():
                existing_target.unlink()

    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_interrogative_prompt_does_not_create_overwrite_action(self, _mock_is_cfg):
        """Question ('Will this overwrite my file?') does NOT create an overwrite plan."""
        existing_target = WORKSPACE_DIR / "existing_doc.txt"
        existing_target.write_text("Existing critical data", encoding="utf-8")

        try:
            chat_resp = self.client.post("/api/v1/chat", json={"message": "Will this overwrite my file 'existing_doc.txt'?"})
            self.assertEqual(chat_resp.status_code, 200)
            data = chat_resp.json()

            self.assertIsNone(data.get("action_id"))
            self.assertEqual(data["status"], "confirmation_required")
            self.assertEqual(data["plan"], [])

            self.assertEqual(existing_target.read_text(encoding="utf-8"), "Existing critical data")
        finally:
            if existing_target.exists():
                existing_target.unlink()

    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_update_noun_does_not_create_overwrite_action(self, _mock_is_cfg):
        """Message containing 'update' as noun does NOT authorize overwriting."""
        existing_target = WORKSPACE_DIR / "existing_doc.txt"
        existing_target.write_text("Existing critical data", encoding="utf-8")

        try:
            chat_resp = self.client.post("/api/v1/chat", json={"message": "Write a status update to 'existing_doc.txt'"})
            self.assertEqual(chat_resp.status_code, 200)
            data = chat_resp.json()

            self.assertIsNone(data.get("action_id"))
            self.assertEqual(data["status"], "confirmation_required")
            self.assertEqual(data["plan"], [])

            self.assertEqual(existing_target.read_text(encoding="utf-8"), "Existing critical data")
        finally:
            if existing_target.exists():
                existing_target.unlink()

    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_quoted_overwrite_does_not_create_overwrite_action(self, _mock_is_cfg):
        """The word 'overwrite' inside quoted text does NOT authorize overwriting."""
        existing_target = WORKSPACE_DIR / "existing_doc.txt"
        existing_target.write_text("Existing critical data", encoding="utf-8")

        try:
            chat_resp = self.client.post(
                "/api/v1/chat",
                json={"message": "Save this quote: 'Never overwrite your principles' into 'existing_doc.txt'"},
            )
            self.assertEqual(chat_resp.status_code, 200)
            data = chat_resp.json()

            self.assertIsNone(data.get("action_id"))
            self.assertEqual(data["status"], "confirmation_required")
            self.assertEqual(data["plan"], [])

            self.assertEqual(existing_target.read_text(encoding="utf-8"), "Existing critical data")
        finally:
            if existing_target.exists():
                existing_target.unlink()

    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_genuine_overwrite_requires_explicit_confirmation_flow(self, _mock_is_cfg):
        """Genuine overwrite request requires explicit separate confirmation before action is planned and approved."""
        existing_target = WORKSPACE_DIR / "existing_doc.txt"
        existing_target.write_text("Old content to replace", encoding="utf-8")

        try:
            # 1. User asks to overwrite: initial turn requires confirmation, does NOT plan action
            chat_resp = self.client.post(
                "/api/v1/chat",
                json={"message": "Please overwrite 'existing_doc.txt' with fresh notes"},
            )
            self.assertEqual(chat_resp.status_code, 200)
            init_data = chat_resp.json()
            self.assertIsNone(init_data.get("action_id"))
            self.assertEqual(init_data["status"], "confirmation_required")
            self.assertEqual(init_data["plan"], [])
            token = init_data["confirmation_token"]
            self.assertIsNotNone(token)
            # Disk is untouched
            self.assertEqual(existing_target.read_text(encoding="utf-8"), "Old content to replace")

            # 2. Separate explicit confirmation via confirm-overwrite endpoint
            confirm_resp = self.client.post(
                "/api/v1/actions/confirm-overwrite",
                json={"confirmation_token": token, "target_file": "existing_doc.txt"},
            )
            self.assertEqual(confirm_resp.status_code, 200)
            plan_data = confirm_resp.json()
            action_id = plan_data["action_id"]
            self.assertIsNotNone(action_id)
            self.assertEqual(plan_data["status"], "pending_approval")
            self.assertTrue(plan_data["plan"][0]["allow_overwrite"])
            self.assertIn("overwrite", plan_data["plan"][0]["description"].lower())
            # Disk is still untouched prior to approval!
            self.assertEqual(existing_target.read_text(encoding="utf-8"), "Old content to replace")

            # 3. Explicit human approval executes safely
            approve_resp = self.client.post(f"/api/v1/actions/{action_id}/approve")
            self.assertEqual(approve_resp.status_code, 200)
            self.assertIn("Hello from KAIROS Web API!", existing_target.read_text(encoding="utf-8"))
        finally:
            if existing_target.exists():
                existing_target.unlink()

    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_confirmation_for_one_path_cannot_authorize_another_path(self, _mock_is_cfg):
        """A confirmation token bound to path A cannot be used to confirm overwriting path B."""
        target_a = WORKSPACE_DIR / "path_a.txt"
        target_b = WORKSPACE_DIR / "path_b.txt"
        target_a.write_text("Target A Content", encoding="utf-8")
        target_b.write_text("Target B Content", encoding="utf-8")

        try:
            chat_resp = self.client.post("/api/v1/chat", json={"message": "Write to 'path_a.txt'"})
            token = chat_resp.json()["confirmation_token"]

            # Attempt to use token for path_b.txt
            mismatch_resp = self.client.post(
                "/api/v1/actions/confirm-overwrite",
                json={"confirmation_token": token, "target_file": "path_b.txt"},
            )
            self.assertEqual(mismatch_resp.status_code, 400)
            self.assertIn("not valid for target file 'path_b.txt'", mismatch_resp.json()["detail"])

            self.assertEqual(target_a.read_text(encoding="utf-8"), "Target A Content")
            self.assertEqual(target_b.read_text(encoding="utf-8"), "Target B Content")
        finally:
            if target_a.exists():
                target_a.unlink()
            if target_b.exists():
                target_b.unlink()

    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_confirmation_token_cannot_be_reused(self, _mock_is_cfg):
        """A confirmation token is single-use and cannot be consumed more than once."""
        target_file = WORKSPACE_DIR / "single_use.txt"
        target_file.write_text("Single use content", encoding="utf-8")

        try:
            chat_resp = self.client.post("/api/v1/chat", json={"message": "Write to 'single_use.txt'"})
            token = chat_resp.json()["confirmation_token"]

            # First consumption succeeds
            r1 = self.client.post(
                "/api/v1/actions/confirm-overwrite",
                json={"confirmation_token": token, "target_file": "single_use.txt"},
            )
            self.assertEqual(r1.status_code, 200)

            # Second consumption with same token fails
            r2 = self.client.post(
                "/api/v1/actions/confirm-overwrite",
                json={"confirmation_token": token, "target_file": "single_use.txt"},
            )
            self.assertEqual(r2.status_code, 400)
            self.assertIn("already been consumed", r2.json()["detail"])
        finally:
            if target_file.exists():
                target_file.unlink()

    def test_confirmation_token_expiry_is_rejected(self):
        """Expired confirmation token cannot create an overwrite plan."""
        from backend.storage import create_overwrite_confirmation
        import uuid

        target_file = WORKSPACE_DIR / "expired_target.txt"
        target_file.write_text("Original text", encoding="utf-8")

        token = "test-expired-token-" + str(uuid.uuid4())
        # Create token that already expired 10 seconds ago
        create_overwrite_confirmation(
            token=token,
            conversation_id=str(uuid.uuid4()),
            target_file="expired_target.txt",
            file_content="Will never write",
            action_desc="Expired test",
            ttl_seconds=-10,
        )

        try:
            resp = self.client.post(
                "/api/v1/actions/confirm-overwrite",
                json={"confirmation_token": token, "target_file": "expired_target.txt"},
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("expired", resp.json()["detail"])
            self.assertEqual(target_file.read_text(encoding="utf-8"), "Original text")
        finally:
            if target_file.exists():
                target_file.unlink()

    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_stage2_post_plan_file_appearance_blocks_execution_safely(self, _mock_is_cfg):
        """If plan was created for non-existing target, a file appearing before approval causes execution to fail safely."""
        target_path = WORKSPACE_DIR / "unexpected_file.txt"
        if target_path.exists():
            target_path.unlink()

        try:
            # 1. Plan created when destination does NOT exist (allow_overwrite=False)
            chat_resp = self.client.post("/api/v1/chat", json={"message": "Write to 'unexpected_file.txt'"})
            self.assertEqual(chat_resp.status_code, 200)
            data = chat_resp.json()
            self.assertFalse(data["plan"][0]["allow_overwrite"])
            action_id = data["action_id"]

            # 2. File appears at destination before approval
            target_path.write_text("Crucial file placed by another process", encoding="utf-8")

            # 3. Attempt approval: must fail with 400, mark failed, and leave file intact
            approve_resp = self.client.post(f"/api/v1/actions/{action_id}/approve")
            self.assertEqual(approve_resp.status_code, 400)
            self.assertIn("overwrite was not authorized", approve_resp.json()["detail"])

            # 4. Check action status in DB
            from backend.storage import get_action
            act = get_action(action_id)
            self.assertEqual(act["status"], "failed")
            self.assertIn("already exists", act["error_message"])

            # 5. Destination file must NOT have been overwritten
            self.assertEqual(target_path.read_text(encoding="utf-8"), "Crucial file placed by another process")
        finally:
            if target_path.exists():
                target_path.unlink()

    @patch("backend.api.is_gemini_configured", return_value=False)
    def test_stage2_rejection_of_overwrite_preserves_existing_file(self, _mock_is_cfg):
        """Rejecting an overwrite plan cancels the action and preserves the original file."""
        existing_target = WORKSPACE_DIR / "protected_target.txt"
        existing_target.write_text("Unchanged secret content", encoding="utf-8")

        try:
            chat_resp = self.client.post("/api/v1/chat", json={"message": "Overwrite 'protected_target.txt'"})
            self.assertEqual(chat_resp.json()["status"], "confirmation_required")
            token = chat_resp.json()["confirmation_token"]

            conf_resp = self.client.post(
                "/api/v1/actions/confirm-overwrite",
                json={"confirmation_token": token, "target_file": "protected_target.txt"},
            )
            action_id = conf_resp.json()["action_id"]

            reject_resp = self.client.post(f"/api/v1/actions/{action_id}/reject")
            self.assertEqual(reject_resp.status_code, 200)
            self.assertEqual(reject_resp.json()["status"], "rejected")

            self.assertEqual(existing_target.read_text(encoding="utf-8"), "Unchanged secret content")
        finally:
            if existing_target.exists():
                existing_target.unlink()

    def test_stage2_legacy_action_without_allow_overwrite_defaults_to_false(self):
        """Legacy stored action without allow_overwrite in steps JSON defaults to denial if target exists."""
        from backend.storage import store_action, get_action
        import uuid

        target_path = WORKSPACE_DIR / "legacy_dest.txt"
        target_path.write_text("Pre-existing data", encoding="utf-8")

        action_id = str(uuid.uuid4())
        conv_id = str(uuid.uuid4())
        # Store action steps with NO allow_overwrite key
        legacy_steps = [
            {
                "step_number": 1,
                "action": "create_file",
                "target_file": "legacy_dest.txt",
                "content": "Malicious or accidental overwrite",
                "description": "Legacy plan step",
            }
        ]
        store_action(
            action_id=action_id,
            conversation_id=conv_id,
            user_message="Legacy request",
            steps=legacy_steps,
        )

        try:
            approve_resp = self.client.post(f"/api/v1/actions/{action_id}/approve")
            self.assertEqual(approve_resp.status_code, 400)
            self.assertIn("already exists and overwrite was not authorized", approve_resp.json()["detail"])

            act = get_action(action_id)
            self.assertEqual(act["status"], "failed")

            # File must be preserved
            self.assertEqual(target_path.read_text(encoding="utf-8"), "Pre-existing data")
        finally:
            if target_path.exists():
                target_path.unlink()


if __name__ == "__main__":
    unittest.main()
