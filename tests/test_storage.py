"""
Unit tests for ECHO SQLite Local Storage, Conversation History & NLP Intent Routing (v0.4)
Tests create, list, load, rename, delete, persistence across restart,
document reference cleanup, and intent validation with zero external dependencies.
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.api import app, clear_pending_actions
from backend.model_client import (
    VALID_INTENTS,
    GemmaPlan,
    parse_and_validate_model_output,
)
from backend.storage import (
    add_message,
    claim_action_for_execution,
    complete_action,
    create_conversation,
    delete_conversation,
    expire_stale_actions,
    fail_action,
    get_action,
    get_conversation,
    get_conversation_context_slice,
    init_db,
    link_document_to_conversation,
    list_conversations,
    recover_interrupted_actions,
    reject_action_record,
    rename_conversation,
    store_action,
    store_document_record,
)


class TestLocalStorageAndHistory(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="echo_test_db_"))
        self.db_path = self.temp_dir / "test_echo.db"
        init_db(self.db_path)
        self.client = TestClient(app)
        clear_pending_actions()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        clear_pending_actions()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_and_get_conversation(self):
        """Conversations can be created and retrieved with metadata."""
        conv = create_conversation(title="Physics Revision", db_path=self.db_path)
        self.assertEqual(conv["title"], "Physics Revision")
        self.assertIn("id", conv)

        loaded = get_conversation(conv["id"], db_path=self.db_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["id"], conv["id"])
        self.assertEqual(loaded["title"], "Physics Revision")
        self.assertEqual(len(loaded["messages"]), 0)

    def test_add_messages_and_auto_title(self):
        """First user message automatically updates default title and stores chronological messages."""
        conv = create_conversation(title="New Chat", db_path=self.db_path)
        msg1 = add_message(
            conversation_id=conv["id"],
            role="user",
            content="Can you explain Newton's second law with formulas?",
            db_path=self.db_path,
        )
        self.assertEqual(msg1["role"], "user")

        msg2 = add_message(
            conversation_id=conv["id"],
            role="assistant",
            content="$F = ma$ where $F$ is force, $m$ is mass, and $a$ is acceleration.",
            metadata={"status": "completed"},
            db_path=self.db_path,
        )

        loaded = get_conversation(conv["id"], db_path=self.db_path)
        self.assertEqual(len(loaded["messages"]), 2)
        self.assertEqual(loaded["messages"][0]["role"], "user")
        self.assertEqual(loaded["messages"][1]["role"], "assistant")
        self.assertIn("Newton's second law", loaded["title"])

    def test_list_and_search_conversations(self):
        """Conversation list orders by recency and supports search filtering."""
        c1 = create_conversation(title="Calculus Exam Questions", db_path=self.db_path)
        c2 = create_conversation(title="Chemistry Lab Report", db_path=self.db_path)

        all_convs = list_conversations(db_path=self.db_path)
        self.assertEqual(len(all_convs), 2)

        # Search for calculus
        search_calc = list_conversations(query="calculus", db_path=self.db_path)
        self.assertEqual(len(search_calc), 1)
        self.assertEqual(search_calc[0]["id"], c1["id"])

        # Search for chem
        search_chem = list_conversations(query="chemistry", db_path=self.db_path)
        self.assertEqual(len(search_chem), 1)
        self.assertEqual(search_chem[0]["id"], c2["id"])

    def test_rename_conversation(self):
        """Conversations can be renamed safely."""
        conv = create_conversation(title="Old Title", db_path=self.db_path)
        renamed = rename_conversation(conv["id"], "Final Exam Preparation", db_path=self.db_path)
        self.assertTrue(renamed)

        loaded = get_conversation(conv["id"], db_path=self.db_path)
        self.assertEqual(loaded["title"], "Final Exam Preparation")

    def test_delete_conversation_removes_messages_and_unreferenced_files(self):
        """Deleting a conversation cleans up messages and only removes unreferenced docs."""
        conv = create_conversation(title="Temporary Session", db_path=self.db_path)
        add_message(conv["id"], "user", "Hello", db_path=self.db_path)

        # Create dummy document file
        doc_file = self.temp_dir / "test_doc.pdf"
        doc_file.write_bytes(b"%PDF-1.4 mock")

        store_document_record(
            doc_id="doc-123",
            filename="test_doc.pdf",
            storage_path=str(doc_file),
            page_count=1,
            size_bytes=14,
            sha256_prefix="123456",
            db_path=self.db_path,
        )
        link_document_to_conversation(conv["id"], "doc-123", db_path=self.db_path)

        # Confirm document exists before delete
        self.assertTrue(doc_file.exists())

        del_res = delete_conversation(conv["id"], db_path=self.db_path)
        self.assertTrue(del_res["deleted"])

        # Check conversation is gone
        self.assertIsNone(get_conversation(conv["id"], db_path=self.db_path))
        # Check unreferenced document file was deleted
        self.assertFalse(doc_file.exists())

    def test_delete_conversation_preserves_shared_document(self):
        """If a document is linked to multiple conversations, deleting one does not delete the file."""
        c1 = create_conversation(title="Conv 1", db_path=self.db_path)
        c2 = create_conversation(title="Conv 2", db_path=self.db_path)

        doc_file = self.temp_dir / "shared_doc.pdf"
        doc_file.write_bytes(b"%PDF-1.4 shared")

        store_document_record(
            doc_id="shared-doc",
            filename="shared_doc.pdf",
            storage_path=str(doc_file),
            page_count=2,
            size_bytes=16,
            sha256_prefix="abcdef",
            db_path=self.db_path,
        )
        link_document_to_conversation(c1["id"], "shared-doc", db_path=self.db_path)
        link_document_to_conversation(c2["id"], "shared-doc", db_path=self.db_path)

        # Delete c1
        delete_conversation(c1["id"], db_path=self.db_path)
        # Shared document file must still exist because c2 references it
        self.assertTrue(doc_file.exists())

        # Now delete c2 -> now file should be removed
        delete_conversation(c2["id"], db_path=self.db_path)
        self.assertFalse(doc_file.exists())

    def test_bounded_conversation_context_slice(self):
        """Context slice yields only recent turns in chronological order."""
        conv = create_conversation(title="Slice Test", db_path=self.db_path)
        for i in range(10):
            add_message(conv["id"], "user" if i % 2 == 0 else "assistant", f"Turn {i}", db_path=self.db_path)

        slice_msgs = get_conversation_context_slice(conv["id"], max_messages=4, db_path=self.db_path)
        self.assertEqual(len(slice_msgs), 4)
        self.assertEqual(slice_msgs[0]["content"], "Turn 6")
        self.assertEqual(slice_msgs[-1]["content"], "Turn 9")

    def test_validated_nlp_intents_in_model_client(self):
        """Model client parses and validates all recognized NLP intents."""
        raw_quiz = """{
            "intent": "practice_quiz",
            "supported": false,
            "assistant_message": "Here is a 5-question quiz on calculus limits:",
            "plan_description": "",
            "file_content": ""
        }"""
        plan_quiz = parse_and_validate_model_output(raw_quiz, model_name="gemma-4-26b-a4b-it")
        self.assertEqual(plan_quiz.intent, "practice_quiz")
        self.assertFalse(plan_quiz.supported)

        raw_map = """{
            "intent": "revision_map",
            "supported": false,
            "assistant_message": "### Revision Map\\n- Unit 1: High Yield [Page 2]",
            "plan_description": "",
            "file_content": ""
        }"""
        plan_map = parse_and_validate_model_output(raw_map, model_name="gemma-4-26b-a4b-it")
        self.assertEqual(plan_map.intent, "revision_map")
        self.assertFalse(plan_map.supported)

        # Invalid intent falls back to unsupported without breaking
        raw_invalid = """{
            "intent": "hack_system_command",
            "supported": false,
            "assistant_message": "I cannot perform that.",
            "plan_description": "",
            "file_content": ""
        }"""
        plan_invalid = parse_and_validate_model_output(raw_invalid, model_name="gemma-4-26b-a4b-it")
        self.assertEqual(plan_invalid.intent, "unsupported")

    def test_api_conversation_endpoints(self):
        """FastAPI endpoints for conversations work seamlessly."""
        # Create
        create_resp = self.client.post("/api/v1/conversations", json={"title": "Math Revision"})
        self.assertEqual(create_resp.status_code, 201)
        conv_id = create_resp.json()["id"]

        # List
        list_resp = self.client.get("/api/v1/conversations")
        self.assertEqual(list_resp.status_code, 200)
        self.assertTrue(any(c["id"] == conv_id for c in list_resp.json()))

        # Chat into this conversation (tested in demo mode without live network calls)
        with patch("backend.api.is_gemini_configured", return_value=False):
            chat_resp = self.client.post(
                "/api/v1/chat",
                json={"message": "Create study notes", "conversation_id": conv_id},
            )
        self.assertEqual(chat_resp.status_code, 200)
        self.assertEqual(chat_resp.json()["conversation_id"], conv_id)

        # Load conversation
        get_resp = self.client.get(f"/api/v1/conversations/{conv_id}")
        self.assertEqual(get_resp.status_code, 200)
        messages = get_resp.json()["messages"]
        self.assertTrue(len(messages) >= 2)

        # Rename
        rename_resp = self.client.patch(
            f"/api/v1/conversations/{conv_id}",
            json={"title": "Updated Math Revision"},
        )
        self.assertEqual(rename_resp.status_code, 200)

        # Delete
        del_resp = self.client.delete(f"/api/v1/conversations/{conv_id}")
        self.assertEqual(del_resp.status_code, 200)

    def test_store_and_get_action(self):
        """Action can be stored durably in SQLite and retrieved."""
        conv = create_conversation(title="Action Chat", db_path=self.db_path)
        steps = [{"step_number": 1, "action": "create_file", "target_file": "demo.txt", "content": "Test content"}]
        stored = store_action(
            action_id="act-123",
            conversation_id=conv["id"],
            user_message="Make file",
            steps=steps,
            mode="demo",
            db_path=self.db_path,
        )
        self.assertEqual(stored["action_id"], "act-123")
        self.assertEqual(stored["status"], "pending_approval")

        fetched = get_action("act-123", db_path=self.db_path)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["action_id"], "act-123")
        self.assertEqual(fetched["status"], "pending_approval")
        self.assertEqual(len(fetched["steps"]), 1)
        self.assertEqual(fetched["steps"][0]["content"], "Test content")

    def test_claim_complete_fail_action_lifecycle(self):
        """Action can be claimed atomically, completed, or failed."""
        conv = create_conversation(title="Claim Chat", db_path=self.db_path)
        store_action(
            action_id="act-claim",
            conversation_id=conv["id"],
            user_message="Do claim",
            steps=[{"step_number": 1, "action": "create_file", "target_file": "demo.txt", "content": "abc"}],
            db_path=self.db_path,
        )

        # 1. Claim succeeds
        claimed, act, reason = claim_action_for_execution("act-claim", db_path=self.db_path)
        self.assertTrue(claimed)
        self.assertEqual(reason, "claimed")
        self.assertEqual(act["status"], "executing")

        # 2. Second concurrent claim fails because status is already 'executing'
        claimed2, act2, reason2 = claim_action_for_execution("act-claim", db_path=self.db_path)
        self.assertFalse(claimed2)
        self.assertEqual(reason2, "executing")

        # 3. Complete action
        success = complete_action("act-claim", db_path=self.db_path)
        self.assertTrue(success)

        completed_act = get_action("act-claim", db_path=self.db_path)
        self.assertEqual(completed_act["status"], "completed")
        self.assertIsNotNone(completed_act["completed_at"])

        # 4. Cannot claim completed action
        claimed3, _, reason3 = claim_action_for_execution("act-claim", db_path=self.db_path)
        self.assertFalse(claimed3)
        self.assertEqual(reason3, "completed")

    def test_failed_action_recovery_behavior(self):
        """Action failing execution enters 'failed' state and can be claimed again for retry."""
        conv = create_conversation(title="Fail Chat", db_path=self.db_path)
        store_action(
            action_id="act-fail",
            conversation_id=conv["id"],
            user_message="Fail then retry",
            steps=[{"step_number": 1, "action": "create_file", "target_file": "demo.txt", "content": "xyz"}],
            db_path=self.db_path,
        )

        claimed, _, _ = claim_action_for_execution("act-fail", db_path=self.db_path)
        self.assertTrue(claimed)

        # Execution fails (e.g. disk error or crash)
        fail_action("act-fail", "Disk write verification failed", db_path=self.db_path)

        failed_act = get_action("act-fail", db_path=self.db_path)
        self.assertEqual(failed_act["status"], "failed")
        self.assertEqual(failed_act["error_message"], "Disk write verification failed")

        # Claiming again for retry succeeds!
        reclaimed, _, reason = claim_action_for_execution("act-fail", db_path=self.db_path)
        self.assertTrue(reclaimed)
        self.assertEqual(reason, "claimed")

    def test_action_rejection(self):
        """Action can be rejected and cannot be claimed subsequently."""
        conv = create_conversation(title="Reject Chat", db_path=self.db_path)
        store_action(
            action_id="act-rej",
            conversation_id=conv["id"],
            user_message="Reject this",
            steps=[],
            db_path=self.db_path,
        )

        rej_success, reason = reject_action_record("act-rej", db_path=self.db_path)
        self.assertTrue(rej_success)
        self.assertEqual(reason, "rejected")

        # Claim fails
        claimed, _, claim_reason = claim_action_for_execution("act-rej", db_path=self.db_path)
        self.assertFalse(claimed)
        self.assertEqual(claim_reason, "rejected")

    def test_expired_action_cannot_be_claimed_or_approved(self):
        """Expired action transitions to 'expired' and cannot be claimed."""
        conv = create_conversation(title="Expire Chat", db_path=self.db_path)
        # Store with negative TTL so it is already expired
        store_action(
            action_id="act-exp",
            conversation_id=conv["id"],
            user_message="Expired action",
            steps=[{"step_number": 1, "action": "create_file", "target_file": "demo.txt", "content": "old"}],
            ttl_seconds=-10,
            db_path=self.db_path,
        )

        # Claim should fail with expired
        claimed, _, reason = claim_action_for_execution("act-exp", db_path=self.db_path)
        self.assertFalse(claimed)
        self.assertEqual(reason, "expired")

        # get_action reflects expired
        act = get_action("act-exp", db_path=self.db_path)
        self.assertEqual(act["status"], "expired")

    def test_conversation_deletion_cascades_to_actions(self):
        """Deleting a conversation cascades to delete associated actions."""
        conv = create_conversation(title="Cascade Chat", db_path=self.db_path)
        store_action(
            action_id="act-cascade",
            conversation_id=conv["id"],
            user_message="Associated action",
            steps=[],
            db_path=self.db_path,
        )

        self.assertIsNotNone(get_action("act-cascade", db_path=self.db_path))

        # Delete conversation
        delete_conversation(conv["id"], db_path=self.db_path)

        # Action is deleted by foreign key cascade
        self.assertIsNone(get_action("act-cascade", db_path=self.db_path))

    def test_schema_migration_preserves_existing_data(self):
        """Schema migration from v1 to v2 creates actions table while preserving existing conversations and messages."""
        # Setup a v1 database without actions table
        old_db_path = self.temp_dir / "old_v1.db"
        from backend.storage import get_connection
        with get_connection(old_db_path) as conn:
            conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);")
            conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (1, '2026-10-01T00:00:00Z');")
            conn.execute("CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);")
            conn.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL);")
            conn.execute("INSERT INTO conversations VALUES ('c1', 'Saved Old Title', '2026-10-01T00:00:00Z', '2026-10-01T00:00:00Z');")
            conn.execute("INSERT INTO messages VALUES ('m1', 'c1', 'user', 'Preserved old content', '{}', '2026-10-01T00:00:00Z');")
            conn.commit()

        # Run init_db on the existing v1 database
        init_db(old_db_path)

        with get_connection(old_db_path) as conn:
            # Verify version is now 3 (latest schema version)
            v_row = conn.execute("SELECT MAX(version) AS ver FROM schema_version;").fetchone()
            self.assertEqual(v_row["ver"], 3)

            # Verify existing data preserved
            c_row = conn.execute("SELECT title FROM conversations WHERE id = 'c1';").fetchone()
            self.assertEqual(c_row["title"], "Saved Old Title")

            m_row = conn.execute("SELECT content FROM messages WHERE id = 'm1';").fetchone()
            self.assertEqual(m_row["content"], "Preserved old content")

            # Verify actions table exists and works
            conn.execute("INSERT INTO actions VALUES ('a1', 'c1', 'pending_approval', 'demo', NULL, 'msg', '[]', '2026-10-01', '2026-10-02', NULL, NULL);")
            conn.commit()

    def test_recover_interrupted_actions_transitions_executing_to_failed(self):
        """
        Actions stranded in 'executing' status during process crash or restart are
        transitioned to 'failed' requiring explicit user retry. They are never automatically re-executed.
        Stranded actions that have passed their expiration deadline become 'expired'.
        """
        conv = create_conversation(title="Crash Recovery Chat", db_path=self.db_path)

        # 1. Unexpired executing action (simulating server crash mid-write)
        store_action(
            action_id="act-crashed-1",
            conversation_id=conv["id"],
            user_message="Create crashed file",
            steps=[{"step_number": 1, "action": "create_file", "target_file": "crashed.txt", "content": "data"}],
            ttl_seconds=3600,
            db_path=self.db_path,
        )
        claim_action_for_execution("act-crashed-1", db_path=self.db_path)

        # 2. Expired executing action (simulating server crashed and stayed offline past TTL)
        store_action(
            action_id="act-crashed-expired",
            conversation_id=conv["id"],
            user_message="Create expired file",
            steps=[{"step_number": 1, "action": "create_file", "target_file": "exp.txt", "content": "data"}],
            ttl_seconds=-10,
            db_path=self.db_path,
        )
        from backend.storage import get_connection
        with get_connection(self.db_path) as conn:
            conn.execute("UPDATE actions SET status = 'executing' WHERE id = 'act-crashed-expired';")
            conn.commit()

        # 3. Normal pending action (should remain untouched)
        store_action(
            action_id="act-normal-pending",
            conversation_id=conv["id"],
            user_message="Pending file",
            steps=[{"step_number": 1, "action": "create_file", "target_file": "pending.txt", "content": "data"}],
            ttl_seconds=3600,
            db_path=self.db_path,
        )

        # 4. Completed action (should remain untouched)
        store_action(
            action_id="act-completed",
            conversation_id=conv["id"],
            user_message="Completed file",
            steps=[],
            ttl_seconds=3600,
            db_path=self.db_path,
        )
        claim_action_for_execution("act-completed", db_path=self.db_path)
        complete_action("act-completed", db_path=self.db_path)

        # Run recovery
        recovered_count = recover_interrupted_actions(db_path=self.db_path)
        self.assertEqual(recovered_count, 1)

        # Verify act-crashed-1 transitioned to failed with descriptive error message
        rec = get_action("act-crashed-1", db_path=self.db_path)
        self.assertEqual(rec["status"], "failed")
        self.assertIn("interrupted by process termination", rec["error_message"])

        # Verify act-crashed-expired transitioned to expired
        rec_exp = get_action("act-crashed-expired", db_path=self.db_path)
        self.assertEqual(rec_exp["status"], "expired")

        # Verify act-normal-pending remained pending_approval
        rec_pend = get_action("act-normal-pending", db_path=self.db_path)
        self.assertEqual(rec_pend["status"], "pending_approval")

        # Verify act-completed remained completed
        rec_comp = get_action("act-completed", db_path=self.db_path)
        self.assertEqual(rec_comp["status"], "completed")

        # Verify failed action can now be claimed and retried cleanly
        claimed_retry, _, reason = claim_action_for_execution("act-crashed-1", db_path=self.db_path)
        self.assertTrue(claimed_retry)
        self.assertEqual(reason, "claimed")

    def test_expire_stale_actions_cleans_overdue_and_preserves_active(self):
        """expire_stale_actions marks overdue pending, executing, and failed actions as expired while preserving unexpired ones."""
        conv = create_conversation(title="Expiry Chat", db_path=self.db_path)

        # Overdue pending
        store_action("act-overdue-pend", conv["id"], "msg", [], ttl_seconds=-10, db_path=self.db_path)
        # Overdue failed
        store_action("act-overdue-fail", conv["id"], "msg", [], ttl_seconds=-10, db_path=self.db_path)
        from backend.storage import get_connection
        with get_connection(self.db_path) as conn:
            conn.execute("UPDATE actions SET status = 'failed' WHERE id = 'act-overdue-fail';")
            conn.commit()

        # Active pending
        store_action("act-active-pend", conv["id"], "msg", [], ttl_seconds=3600, db_path=self.db_path)

        expired_count = expire_stale_actions(db_path=self.db_path)
        self.assertEqual(expired_count, 2)

        self.assertEqual(get_action("act-overdue-pend", db_path=self.db_path)["status"], "expired")
        self.assertEqual(get_action("act-overdue-fail", db_path=self.db_path)["status"], "expired")
        self.assertEqual(get_action("act-active-pend", db_path=self.db_path)["status"], "pending_approval")

    def test_get_db_path_fresh_install_defaults_to_kairos(self):
        """On fresh install when neither database exists, get_db_path defaults to kairos_history.db."""
        from backend.storage import get_db_path
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            with patch("backend.storage.get_data_dir", return_value=tmppath):
                resolved = get_db_path()
                self.assertEqual(resolved, tmppath / "kairos_history.db")
                self.assertFalse(resolved.exists())

    def test_get_db_path_legacy_compatibility_when_only_echo_exists(self):
        """When only echo_history.db exists in the data directory, get_db_path preserves legacy access."""
        from backend.storage import get_db_path
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            legacy_db = tmppath / "echo_history.db"
            legacy_db.touch()
            with patch("backend.storage.get_data_dir", return_value=tmppath):
                resolved = get_db_path()
                self.assertEqual(resolved, legacy_db)

    def test_get_db_path_deterministic_precedence_when_both_exist(self):
        """When both kairos_history.db and echo_history.db exist, kairos_history.db takes precedence without modifying either."""
        from backend.storage import get_db_path
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            kairos_db = tmppath / "kairos_history.db"
            echo_db = tmppath / "echo_history.db"
            kairos_db.write_text("kairos-db", encoding="utf-8")
            echo_db.write_text("echo-db", encoding="utf-8")
            with patch("backend.storage.get_data_dir", return_value=tmppath):
                resolved = get_db_path()
                self.assertEqual(resolved, kairos_db)
                # Ensure neither file was deleted or overwritten
                self.assertEqual(kairos_db.read_text(encoding="utf-8"), "kairos-db")
                self.assertEqual(echo_db.read_text(encoding="utf-8"), "echo-db")

    def test_get_db_path_when_only_kairos_db_exists(self):
        """When only kairos_history.db exists, it is selected cleanly."""
        from backend.storage import get_db_path
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            kairos_db = tmppath / "kairos_history.db"
            kairos_db.touch()
            with patch("backend.storage.get_data_dir", return_value=tmppath):
                resolved = get_db_path()
                self.assertEqual(resolved, kairos_db)


if __name__ == "__main__":
    unittest.main()
