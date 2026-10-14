"""
Unit tests for ECHO Agent (Phase 1)
Tests core safety guarantees and the permission workflow:
  1. Rejection test: rejecting a plan creates no files.
  2. Approval test: approving a plan creates the expected file and content.
  3. Security test: attempts to escape workspace/ are blocked with PermissionError.
  4. Cleanup: ensures all created test files are removed, even on test failure.
"""

from pathlib import Path
import sys
import unittest

# Ensure the project root is in the Python search path so backend can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent import EchoAgent, resolve_safe_path, WORKSPACE_DIR


class TestEchoAgent(unittest.TestCase):

    def setUp(self):
        """Runs BEFORE every test: define target file and ensure clean state."""
        self.target_file = WORKSPACE_DIR / "demo.txt"

        # Clean up any leftover file before the test begins
        if self.target_file.exists():
            self.target_file.unlink()

        # Register cleanup to run AFTER every test, even if the test fails
        self.addCleanup(self.cleanup_target_file)

    def cleanup_target_file(self):
        """Cleanup helper: safely removes test files from workspace/."""
        if self.target_file.exists():
            self.target_file.unlink()

    # --------------------------------------------------------------------------
    # Test 1: Rejection workflow
    # --------------------------------------------------------------------------
    def test_rejection_does_not_create_file(self):
        """
        Test that when the user rejects the plan:
          - agent.run returns False
          - workspace/demo.txt is NEVER created
        """
        agent = EchoAgent()
        result = agent.run("Create a demo file in my workspace", auto_approve=False)

        self.assertFalse(result, "agent.run should return False when plan is rejected")
        self.assertFalse(
            self.target_file.exists(),
            "workspace/demo.txt should NOT exist after rejection"
        )

    # --------------------------------------------------------------------------
    # Test 2: Approval workflow
    # --------------------------------------------------------------------------
    def test_approval_creates_expected_file_and_content(self):
        """
        Test that when the user approves the plan:
          - agent.run returns True
          - workspace/demo.txt is created
          - The file contains the expected greeting message
        """
        agent = EchoAgent()
        result = agent.run("Create a demo file in my workspace", auto_approve=True)

        self.assertTrue(result, "agent.run should return True when plan is approved")
        self.assertTrue(
            self.target_file.exists(),
            "workspace/demo.txt should exist after approval"
        )

        content = self.target_file.read_text(encoding="utf-8")
        self.assertTrue("Hello from KAIROS!" in content or "Hello from ECHO!" in content)
        self.assertIn("Action performed safely inside workspace/.", content)
        self.assertIn("Permission was explicitly granted by user.", content)

    # --------------------------------------------------------------------------
    # Test 3: Path containment security (Sandboxing)
    # --------------------------------------------------------------------------
    def test_resolve_safe_path_blocks_directory_escape(self):
        """
        Test that resolve_safe_path rejects any attempts to escape workspace/,
        raising PermissionError for path traversal and absolute system paths.
        """
        escape_attempts = [
            "../../etc/passwd",
            "../backend/agent.py",
            "/etc/shadow",
            "../../../secret.txt",
        ]

        for malicious_path in escape_attempts:
            with self.subTest(path=malicious_path):
                with self.assertRaises(
                    PermissionError,
                    msg=f"Should block escape attempt: {malicious_path}"
                ):
                    resolve_safe_path(malicious_path)

    # --------------------------------------------------------------------------
    # Test 4: Valid path resolves correctly inside workspace
    # --------------------------------------------------------------------------
    def test_resolve_safe_path_allows_workspace_files(self):
        """
        Test that legitimate files inside workspace/ resolve correctly
        without raising errors.
        """
        safe_path = resolve_safe_path("demo.txt")
        expected_path = (WORKSPACE_DIR / "demo.txt").resolve()

        self.assertEqual(safe_path, expected_path)
        self.assertTrue(safe_path.is_relative_to(WORKSPACE_DIR.resolve()))

    # --------------------------------------------------------------------------
    # Stage 2 Tests: Generalized File Targets, Symlinks, Nested Paths & Overwrites
    # --------------------------------------------------------------------------
    def test_stage2_valid_root_and_nested_filenames(self):
        """Valid root and nested paths resolve cleanly inside workspace."""
        from backend.agent import validate_and_resolve_target_path
        root_target = validate_and_resolve_target_path("summary.txt")
        self.assertEqual(root_target, (WORKSPACE_DIR / "summary.txt").resolve())

        nested_target = validate_and_resolve_target_path("notes/deep/summary.md")
        self.assertEqual(nested_target, (WORKSPACE_DIR / "notes/deep/summary.md").resolve())

    def test_stage2_rejects_empty_and_whitespace_paths(self):
        """Empty or whitespace target paths raise ValueError."""
        from backend.agent import validate_and_resolve_target_path
        with self.assertRaises(ValueError):
            validate_and_resolve_target_path("")
        with self.assertRaises(ValueError):
            validate_and_resolve_target_path("   \n\t  ")

    def test_stage2_rejects_absolute_paths_and_traversal(self):
        """Absolute paths, null bytes, and traversal tokens raise PermissionError."""
        from backend.agent import validate_and_resolve_target_path
        bad_paths = [
            "/etc/passwd",
            "/tmp/test.txt",
            "notes/../../etc/shadow",
            "subdir/../../../escape.txt",
            "file\0name.txt",
        ]
        for p in bad_paths:
            with self.subTest(path=p):
                with self.assertRaises(PermissionError):
                    validate_and_resolve_target_path(p)

    def test_stage2_rejects_symlink_escapes(self):
        """Symlinks pointing outside workspace directory are detected and rejected."""
        from backend.agent import validate_and_resolve_target_path
        import tempfile
        # Create an external directory and link a directory inside workspace to it
        with tempfile.TemporaryDirectory() as external_dir:
            ext_path = Path(external_dir)
            symlink_in_workspace = WORKSPACE_DIR / "evil_symlink"
            if symlink_in_workspace.is_symlink() or symlink_in_workspace.exists():
                symlink_in_workspace.unlink()

            try:
                symlink_in_workspace.symlink_to(ext_path, target_is_directory=True)
                with self.assertRaises(PermissionError):
                    validate_and_resolve_target_path("evil_symlink/secret.txt")
            finally:
                if symlink_in_workspace.is_symlink() or symlink_in_workspace.exists():
                    symlink_in_workspace.unlink()

    def test_stage2_safe_atomic_write_nested_and_overwrite(self):
        """Atomic write creates parent directories and supports explicit overwrite."""
        from backend.agent import safe_atomic_write
        nested_rel = "test_run/nested/doc.txt"
        nested_file = WORKSPACE_DIR / nested_rel
        if nested_file.exists():
            nested_file.unlink()
        if nested_file.parent.exists():
            import shutil
            shutil.rmtree(nested_file.parent)

        try:
            # 1. Write creates nested directories automatically
            written_path = safe_atomic_write(nested_rel, "Initial version")
            self.assertTrue(written_path.exists())
            self.assertEqual(written_path.read_text(encoding="utf-8"), "Initial version")

            # 2. Overwrite check: allow_overwrite=False raises FileExistsError
            with self.assertRaises(FileExistsError):
                safe_atomic_write(nested_rel, "Second version", allow_overwrite=False)

            # 3. Explicit overwrite succeeds
            safe_atomic_write(nested_rel, "Updated version", allow_overwrite=True)
            self.assertEqual(written_path.read_text(encoding="utf-8"), "Updated version")
        finally:
            if (WORKSPACE_DIR / "test_run").exists():
                import shutil
                shutil.rmtree(WORKSPACE_DIR / "test_run")

    def test_safe_atomic_write_blocks_post_creation_appearance(self):
        """If target file appears right before atomic replace when allow_overwrite=False, it raises FileExistsError."""
        from backend.agent import safe_atomic_write
        import tempfile
        from unittest.mock import patch

        rel_target = "racy_target.txt"
        target_path = WORKSPACE_DIR / rel_target
        if target_path.exists():
            target_path.unlink()

        original_named_temp = tempfile.NamedTemporaryFile

        def mock_named_temp(*args, **kwargs):
            tf = original_named_temp(*args, **kwargs)
            # Simulate a file appearing right during temp file preparation
            target_path.write_text("Intervening file content", encoding="utf-8")
            return tf

        try:
            with patch("tempfile.NamedTemporaryFile", side_effect=mock_named_temp):
                with self.assertRaises(FileExistsError) as ctx:
                    safe_atomic_write(rel_target, "New content", allow_overwrite=False)
                self.assertIn("appeared before approval and overwrite was not authorized", str(ctx.exception))

            # Ensure existing intervening file is preserved and not replaced
            self.assertEqual(target_path.read_text(encoding="utf-8"), "Intervening file content")
        finally:
            if target_path.exists():
                target_path.unlink()

    def test_safe_atomic_write_blocks_destination_appearing_before_final_installation(self):
        """
        Stage 2A Regression Test:
        Simulates the destination file appearing immediately before the final installation
        operation (os.link), bypassing the preceding exists() checks.
        Asserts that the atomic no-clobber link operation fails safely with FileExistsError,
        the competing file's content remains completely unchanged, and temporary files are cleaned up.
        """
        from backend.agent import safe_atomic_write
        from unittest.mock import patch
        import os

        rel_target = "toctou_race_target.txt"
        target_path = WORKSPACE_DIR / rel_target
        if target_path.exists():
            target_path.unlink()

        real_os_link = os.link

        def simulate_racing_destination_creation(src, dst, *args, **kwargs):
            # Destination appears concurrently right before os.link is called
            target_path.write_text("COMPETING PROCESS DATA - DO NOT OVERWRITE", encoding="utf-8")
            # Invoke the real os.link; the filesystem must reject it atomically with EEXIST
            return real_os_link(src, dst, *args, **kwargs)

        try:
            with patch("os.link", side_effect=simulate_racing_destination_creation):
                with self.assertRaises(FileExistsError) as ctx:
                    safe_atomic_write(rel_target, "NEW ATTEMPTED CONTENT", allow_overwrite=False)
                self.assertIn("overwrite was not authorized", str(ctx.exception))

            # Verify the competing file was NOT overwritten or clobbered
            self.assertEqual(
                target_path.read_text(encoding="utf-8"),
                "COMPETING PROCESS DATA - DO NOT OVERWRITE"
            )

            # Verify temporary files were cleaned up
            tmp_files = list(WORKSPACE_DIR.glob(".tmp_toctou_race_target.txt_*"))
            self.assertEqual(tmp_files, [], "Temporary files must be cleaned up on failure")
        finally:
            if target_path.exists():
                target_path.unlink()
            for tmp in WORKSPACE_DIR.glob(".tmp_toctou_race_target.txt_*"):
                tmp.unlink()

    def test_safe_atomic_write_overwrite_allowed_replaces_existing_file(self):
        """When allow_overwrite=True, safe_atomic_write replaces existing destination atomically."""
        from backend.agent import safe_atomic_write

        rel_target = "allowed_overwrite_target.txt"
        target_path = WORKSPACE_DIR / rel_target
        target_path.write_text("Old content before explicit overwrite", encoding="utf-8")

        try:
            written_path = safe_atomic_write(
                rel_target, "New authorized overwrite content", allow_overwrite=True
            )
            self.assertEqual(written_path, target_path)
            self.assertEqual(
                target_path.read_text(encoding="utf-8"),
                "New authorized overwrite content"
            )
            # Ensure no temporary files left behind
            tmp_files = list(WORKSPACE_DIR.glob(".tmp_allowed_overwrite_target.txt_*"))
            self.assertEqual(tmp_files, [], "Temporary files must be cleaned up after successful overwrite")
        finally:
            if target_path.exists():
                target_path.unlink()
            for tmp in WORKSPACE_DIR.glob(".tmp_allowed_overwrite_target.txt_*"):
                tmp.unlink()

    def test_safe_atomic_write_fails_closed_when_no_clobber_unavailable(self):
        """If atomic no-clobber (os.link) is unavailable (e.g. filesystem error), fail closed without falling back to clobbering replace."""
        from backend.agent import safe_atomic_write
        from unittest.mock import patch
        import errno

        rel_target = "fail_closed_target.txt"
        target_path = WORKSPACE_DIR / rel_target
        if target_path.exists():
            target_path.unlink()

        def mock_unsupported_link(src, dst, *args, **kwargs):
            raise OSError(errno.EOPNOTSUPP, "Operation not supported on filesystem")

        try:
            with patch("os.link", side_effect=mock_unsupported_link):
                with self.assertRaises(OSError) as ctx:
                    safe_atomic_write(rel_target, "Payload", allow_overwrite=False)
                self.assertIn("Atomic non-overwriting installation failed", str(ctx.exception))

            # Target file was NOT created via fallback
            self.assertFalse(target_path.exists())

            # Temporary files were cleaned up
            tmp_files = list(WORKSPACE_DIR.glob(".tmp_fail_closed_target.txt_*"))
            self.assertEqual(tmp_files, [], "Temporary file must be cleaned up on fail-closed error")
        finally:
            if target_path.exists():
                target_path.unlink()
            for tmp in WORKSPACE_DIR.glob(".tmp_fail_closed_target.txt_*"):
                tmp.unlink()


if __name__ == "__main__":
    unittest.main()
