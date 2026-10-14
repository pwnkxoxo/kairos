"""
KAIROS Prototype - Phase 1
A minimal, permission-controlled AI assistant workflow:
  1. Understand  -> Parse the user's intent.
  2. Plan        -> Formulate explicit, structured steps.
  3. Approval    -> Pause execution and ask human for permission.
  4. Act         -> Execute safe actions ONLY inside workspace/ (No shell commands).
  5. Verify      -> Check that the expected changes were made correctly.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional


# ------------------------------------------------------------------------------
# 1. Security Configuration: Sandboxed Workspace
# ------------------------------------------------------------------------------
# All file actions are strictly constrained to this directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"


def resolve_safe_path(filename: str) -> Path:
    """
    Ensure the target file path stays strictly inside the workspace directory.
    Rejects empty, absolute, or traversing paths (e.g., '../../etc/passwd').
    Also checks for symlink escapes where intermediate components or existing targets
    point outside WORKSPACE_DIR.
    """
    return validate_and_resolve_target_path(filename)


def validate_and_resolve_target_path(filename: str) -> Path:
    """
    Validates a target filename or relative path and resolves it safely within WORKSPACE_DIR.
    Guarantees:
      - Path is non-empty and non-whitespace.
      - Reject absolute paths (e.g. '/etc/passwd', 'C:\\windows').
      - Rejects null bytes and traversal tokens.
      - Canonical destination (.resolve()) is strictly relative to WORKSPACE_DIR.resolve().
      - Symlink checks: If any existing ancestor or the file itself is a symlink pointing outside,
        PermissionError is raised.
    """
    if not filename or not filename.strip():
        raise ValueError("Target filename cannot be empty or whitespace.")

    clean_name = filename.strip()
    if "\0" in clean_name:
        raise PermissionError("Security Alert: Null byte detected in target path!")

    raw_path = Path(clean_name)
    if raw_path.is_absolute():
        raise PermissionError(f"Security Alert: Absolute paths are strictly forbidden: '{clean_name}'")

    canonical_workspace = WORKSPACE_DIR.resolve()

    # Step-by-step symlink and containment validation across path parts
    cur = canonical_workspace
    for part in raw_path.parts:
        if part in ("..", "/", "\\"):
            pass  # Path resolution will evaluate traversal, but check strictly
        cur = (cur / part).resolve()
        if not cur.is_relative_to(canonical_workspace):
            raise PermissionError(
                f"Security Alert: Path component '{part}' escapes the safe workspace directory!"
            )

    target = (WORKSPACE_DIR / clean_name).resolve()
    if not target.is_relative_to(canonical_workspace):
        raise PermissionError(
            f"Security Alert: Target path '{target}' is outside the safe workspace!"
        )

    # Symlink escape verification if the target or any parent exists
    check_node = target
    while check_node != canonical_workspace and check_node != check_node.parent:
        if check_node.is_symlink():
            resolved_link = check_node.resolve()
            if not resolved_link.is_relative_to(canonical_workspace):
                raise PermissionError(
                    f"Security Alert: Symlink '{check_node}' points outside safe workspace!"
                )
        check_node = check_node.parent

    return target


def safe_atomic_write(filename: str, content: str, allow_overwrite: bool = False) -> Path:
    """
    Safely writes content to target file within WORKSPACE_DIR using atomic replacement.
    Revalidates canonical path containment before write and before atomic installation.
    Writes first to a temporary file in WORKSPACE_DIR with flush and fsync.
    When allow_overwrite=False, installs using atomic non-clobber hard link (os.link),
    preventing TOCTOU overwrite race conditions on POSIX systems.
    When allow_overwrite=True, replaces the destination atomically (os.replace).
    Ensures temporary files are cleaned up on all success and failure paths.
    """
    # 1. Revalidate target path containment
    target_path = validate_and_resolve_target_path(filename)
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. Revalidate directory containment
    canonical_workspace = WORKSPACE_DIR.resolve()
    if not target_dir.resolve().is_relative_to(canonical_workspace):
        raise PermissionError(
            f"Security Alert: Target directory '{target_dir}' is outside the safe workspace!"
        )

    # 3. Check initial overwrite policy
    if target_path.exists() and not allow_overwrite:
        raise FileExistsError(f"Target file '{filename}' already exists and overwrite was not authorized.")

    # 4. Create temporary file in same directory (ensures same filesystem for atomic operations)
    import os
    import tempfile

    temp_path: Optional[Path] = None
    try:
        prefix = f".tmp_{target_path.name}_"
        with tempfile.NamedTemporaryFile("w", dir=target_dir, prefix=prefix, delete=False, encoding="utf-8") as tf:
            temp_path = Path(tf.name)
            tf.write(content)
            tf.flush()
            os.fsync(tf.fileno())

        # 5. Final containment check before installation
        if not temp_path.resolve().is_relative_to(canonical_workspace):
            raise PermissionError("Security Alert: Temporary file escaped workspace!")

        # 6. Re-check overwrite authorization before installation
        # If a file appeared concurrently or after initial check and overwrite was not granted, abort safely.
        if target_path.exists() and not allow_overwrite:
            raise FileExistsError(f"Target file '{filename}' appeared before approval and overwrite was not authorized.")

        # 7. Atomic installation
        if not allow_overwrite:
            # Atomic no-clobber operation using hard link on same filesystem.
            # If target_path exists or appeared concurrently, link() fails with FileExistsError (EEXIST) atomically.
            try:
                os.link(temp_path, target_path)
            except FileExistsError:
                raise FileExistsError(
                    f"Target file '{filename}' appeared before approval and overwrite was not authorized."
                )
            except OSError as exc:
                # If a safe no-clobber operation is unavailable, fail closed instead of falling back to an overwriting rename.
                raise OSError(
                    f"Atomic non-overwriting installation failed for '{filename}': {exc}"
                ) from exc
        else:
            # Explicit overwrite authorized: atomic replacement via os.replace
            temp_path.replace(target_path)

        return target_path
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# ------------------------------------------------------------------------------
# 2. Data Models: What a Plan Looks Like
# ------------------------------------------------------------------------------
@dataclass
class PlanStep:
    step_number: int
    action: str          # e.g., "create_file"
    target_file: str     # relative to workspace/, e.g., "demo.txt"
    content: str         # content to write
    description: str     # plain English explanation for the user
    allow_overwrite: bool = False



# ------------------------------------------------------------------------------
# 3. Core KAIROS Workflow Engine
# ------------------------------------------------------------------------------
class KairosAgent:
    def __init__(self, workspace_dir: Path = WORKSPACE_DIR):
        self.workspace_dir = workspace_dir
        # Ensure the workspace directory exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def understand(self, user_request: str) -> str:
        """Step 1: Understand user request."""
        print(f"\n[1. UNDERSTAND] Received user request:")
        print(f"    \"{user_request}\"")
        return user_request.strip()

    def plan(self, user_request: str) -> List[PlanStep]:
        """
        Step 2: Plan
        Creates a list of explicit steps.
        In this prototype, extracts requested filename if present (or defaults to demo.txt).
        Validates target path against safe workspace rules.
        """
        print(f"\n[2. PLAN] Generating execution plan...")

        # Check if user specified a file path in quotes or following common keywords
        filename = "demo.txt"
        import re
        match = re.search(r"(?:into|to|file|named|name)\s+[`'\"]([a-zA-Z0-9_\-\./\\]+)[`'\"]", user_request, re.IGNORECASE)
        if not match:
            match = re.search(r"[`'\"]([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)[`'\"]", user_request)
        if not match:
            # Word-based pattern: e.g. "to notes/summary.txt"
            match = re.search(r"(?:into|to|file)\s+([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)", user_request, re.IGNORECASE)

        if match:
            candidate = match.group(1).replace("\\", "/").strip()
            # Validate candidate path
            try:
                validate_and_resolve_target_path(candidate)
                filename = candidate
            except (ValueError, PermissionError):
                filename = "demo.txt"

        content = (
            "========================================\n"
            "Hello from KAIROS!\n"
            "Action performed safely inside workspace/.\n"
            "Permission was explicitly granted by user.\n"
            "========================================\n"
        )

        steps = [
            PlanStep(
                step_number=1,
                action="create_file",
                target_file=filename,
                content=content,
                description=f"Create a harmless text file '{filename}' with greeting content inside workspace/.",
            )
        ]

        for step in steps:
            print(f"    Step {step.step_number}: [{step.action}] {step.description}")
            print(f"            Target: workspace/{step.target_file}")

        return steps

    def ask_for_approval(self, steps: List[PlanStep], auto_approve: Optional[bool] = None) -> bool:
        """
        Step 3: Ask for Approval
        Execution stops until the human user gives permission.
        """
        print(f"\n[3. APPROVAL] KAIROS is requesting permission to execute {len(steps)} step(s).")
        print("    ⚠️  No action has been taken yet.")

        # If a flag was provided (for automated testing/scripts), use that
        if auto_approve is not None:
            if auto_approve:
                print("    Decision: APPROVED (automated input)")
                return True
            else:
                print("    Decision: REJECTED (automated input)")
                return False

        # Otherwise, ask interactively via standard input
        try:
            choice = input("    Do you approve this plan? (yes/no): ").strip().lower()
            return choice in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            print("\n    Input interrupted. Defaulting to rejection for safety.")
            return False

    def act(self, steps: List[PlanStep]) -> List[dict]:
        """
        Step 4: Act
        Execute the approved steps safely using atomic writes.
        SECURITY GUARANTEES:
        - NEVER executes arbitrary shell commands.
        - NEVER modifies files outside the designated workspace/ directory.
        """
        print(f"\n[4. ACT] Executing approved steps...")
        results = []

        for step in steps:
            if step.action == "create_file":
                # Validate and atomic write strictly within workspace
                safe_path = safe_atomic_write(step.target_file, step.content, allow_overwrite=step.allow_overwrite)
                print(f"    Executing Step {step.step_number}: Writing to {safe_path.name}...")

                results.append({
                    "step_number": step.step_number,
                    "target_file": safe_path,
                    "status": "written",
                })
            else:
                print(f"    ⚠️ Unknown or unsupported action '{step.action}'. Skipped for safety.")
                results.append({
                    "step_number": step.step_number,
                    "status": "skipped",
                })

        return results

    def verify(self, results: List[dict]) -> bool:
        """
        Step 5: Verify
        Confirm that the actions succeeded on disk and report back.
        """
        print(f"\n[5. VERIFY] Verifying results on disk...")
        all_passed = True

        for res in results:
            if res.get("status") == "written":
                target_path = res["target_file"]
                if target_path.exists() and target_path.stat().st_size > 0:
                    print(f"    ✅ Verified: {target_path.name} exists and is {target_path.stat().st_size} bytes.")
                else:
                    print(f"    ❌ Verification Failed: {target_path.name} was not found or is empty!")
                    all_passed = False

        if all_passed:
            print("\n🎉 All steps verified successfully. Task complete!")
        else:
            print("\n⚠️ Verification failed for one or more steps.")

        return all_passed

    def run(self, user_request: str, auto_approve: Optional[bool] = None) -> bool:
        """
        Full 5-step loop orchestration:
        Understand -> Plan -> Ask for approval -> Act -> Verify.
        """
        print("=" * 60)
        print(" KAIROS: The Permission-Controlled AI Assistant ")
        print("=" * 60)

        # 1. Understand
        self.understand(user_request)

        # 2. Plan
        steps = self.plan(user_request)

        # 3. Ask for Approval
        approved = self.ask_for_approval(steps, auto_approve=auto_approve)

        if not approved:
            print("\n🛑 Execution rejected by user. No files were created or modified.")
            print("=" * 60)
            return False

        # 4. Act
        results = self.act(steps)

        # 5. Verify
        success = self.verify(results)
        print("=" * 60)
        return success


# Backward-compatibility alias
EchoAgent = KairosAgent


# ------------------------------------------------------------------------------
# 4. Command Line Entry Point
# ------------------------------------------------------------------------------
def main():
    agent = KairosAgent()
    user_request = "Create a demo file in my workspace"

    # Support optional flags for automated testing:
    #   python backend/agent.py --approve
    #   python backend/agent.py --reject
    auto_approve: Optional[bool] = None
    if "--approve" in sys.argv:
        auto_approve = True
    elif "--reject" in sys.argv:
        auto_approve = False

    agent.run(user_request, auto_approve=auto_approve)


if __name__ == "__main__":
    main()
