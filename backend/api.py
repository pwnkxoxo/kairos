"""
KAIROS Web API - Phase 3
FastAPI backend providing permission-controlled endpoints with hosted Gemma 4 AI:
  - GET  /health                       -> Health check and model mode status
  - POST /api/v1/chat                  -> Generates a safe plan (via Gemma 4 or safe demo mode)
  - POST /api/v1/actions/{id}/approve  -> Approves, executes, and verifies the plan
  - POST /api/v1/actions/{id}/reject   -> Rejects the plan without executing any actions
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.agent import WORKSPACE_DIR, resolve_safe_path, safe_atomic_write
from backend.model_client import (
    ModelClientError,
    analyze_pdf_content,
    generate_gemma_plan,
    get_model_name,
    is_gemini_configured,
    synthesize_web_search,
)
from backend.search_service import (
    SearchServiceError,
    answer_datetime_query,
    execute_tavily_search,
    format_search_evidence,
    is_direct_datetime_query,
    is_tavily_configured,
    should_search_web,
)
from backend.weather_service import (
    handle_weather_query,
    is_weather_query,
)
from backend.pdf_service import (
    MAX_IMAGE_BYTES,
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
    PDF_STORAGE_DIR,
    SUPPORTED_IMAGE_TYPES,
    clear_pdf_registry,
    detect_file_type,
    get_doc_metadata,
    pdf_registry,
    prepare_model_context,
    validate_and_store_image,
    validate_and_store_pdf,
)
from backend.url_service import (
    MAX_URL_FETCH_BYTES,
    MAX_URL_REDIRECTS,
    SsrfBlockedError,
    UrlContentTooLargeError,
    UrlFetchError,
    UrlUnsupportedContentTypeError,
    UrlValidationError,
    extract_html_content,
    safe_fetch_url,
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
    get_conversation_documents,
    link_document_to_conversation,
    list_conversations,
    recover_interrupted_actions,
    reject_action_record,
    rename_conversation,
    store_action,
    store_document_record,
)

# ------------------------------------------------------------------------------
# 1. FastAPI Application Initialization
# ------------------------------------------------------------------------------
app = FastAPI(
    title="KAIROS API",
    description="A permission-controlled AI assistant backend powered by Gemma 4",
    version="0.5.0",
)


@app.on_event("startup")
def on_startup() -> None:
    """
    On backend startup, recover any stranded 'executing' actions left behind by an
    unclean process exit or crash, transitioning them to 'failed'.
    Also marks any stale actions that expired while the server was offline as 'expired'.
    Never re-executes actions automatically.
    """
    recover_interrupted_actions()
    expire_stale_actions()


# ------------------------------------------------------------------------------
# 2. Action Store & Backward-Compatibility Helper
# ------------------------------------------------------------------------------
# Actions are now persisted durably in SQLite.
# clear_pending_actions() is maintained for backward-compatibility with existing tests.
def clear_pending_actions() -> None:
    """Helper used primarily by unit tests to clear stored actions and overwrite confirmations."""
    from backend.storage import get_connection
    with get_connection() as conn:
        conn.execute("DELETE FROM actions;")
        conn.execute("DELETE FROM overwrite_confirmations;")
        conn.commit()


# ------------------------------------------------------------------------------
# 3. Pydantic Request & Response Schemas
# ------------------------------------------------------------------------------
class ConversationCreateRequest(BaseModel):
    title: Optional[str] = Field(default="New Chat", max_length=150)


class ConversationRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=150)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User instruction or question for KAIROS.")
    conversation_id: Optional[str] = Field(default=None, description="Optional active conversation ID for local memory.")
    confirmation_token: Optional[str] = Field(default=None, description="Optional server-issued overwrite confirmation token.")


class ConfirmOverwriteRequest(BaseModel):
    confirmation_token: str = Field(..., min_length=1, description="Server-issued confirmation token for authorizing file overwrite.")
    target_file: Optional[str] = Field(default=None, description="Optional expected target file for path verification.")


class PlanStepSchema(BaseModel):
    step_number: int
    action: str
    target_file: str
    description: str
    allow_overwrite: bool = False


class ChatResponse(BaseModel):
    action_id: Optional[str] = None
    conversation_id: Optional[str] = None
    status: str  # 'pending_approval', 'confirmation_required', or 'unsupported'
    message: str
    ai_disclaimer: str
    supported: bool = True
    intent: Optional[str] = "create_file"
    mode: str = "demo"  # 'gemma' or 'demo'
    plan: List[PlanStepSchema] = []
    file_preview: Optional[str] = None
    confirmation_token: Optional[str] = None
    confirmation_target_file: Optional[str] = None


class ActionResult(BaseModel):
    step_number: int
    target_file: str
    status: str
    bytes_written: int


class ApprovalResponse(BaseModel):
    action_id: str
    status: str
    message: str
    results: List[ActionResult]


class RejectionResponse(BaseModel):
    action_id: str
    status: str
    message: str


# ------------------------------------------------------------------------------
# 4. API Endpoints
# ------------------------------------------------------------------------------
@app.get("/health", summary="Health Check")
def health_check() -> Dict[str, str]:
    """Check that the KAIROS backend service is alive and report mode."""
    gemma_active = is_gemini_configured()
    return {
        "status": "healthy",
        "service": "KAIROS",
        "version": "0.5.0",
        "mode": "gemma" if gemma_active else "demo",
        "model": get_model_name() if gemma_active else "demo_template",
    }


# ------------------------------------------------------------------------------
# Conversation History Endpoints (SQLite Persistent Memory)
# ------------------------------------------------------------------------------
@app.get("/api/v1/conversations", summary="List conversations")
def get_conversations(q: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Lists locally saved conversations ordered by recency, optional search query."""
    return list_conversations(query=q, limit=limit)


@app.post("/api/v1/conversations", summary="Create a new conversation", status_code=status.HTTP_201_CREATED)
def new_conversation(req: Optional[ConversationCreateRequest] = None) -> Dict[str, Any]:
    """Creates a new conversation record."""
    title = req.title if req and req.title else "New Chat"
    return create_conversation(title=title)


@app.get("/api/v1/conversations/{conv_id}", summary="Get conversation messages and documents")
def fetch_conversation(conv_id: str) -> Dict[str, Any]:
    """Loads an existing conversation, its message history, and attached documents."""
    conv = get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return conv


@app.patch("/api/v1/conversations/{conv_id}", summary="Rename conversation")
def update_conversation(conv_id: str, req: ConversationRenameRequest) -> Dict[str, Any]:
    """Renames an existing conversation title."""
    success = rename_conversation(conv_id, req.title)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found or invalid title.")
    return {"id": conv_id, "title": req.title.strip()}


@app.delete("/api/v1/conversations/{conv_id}", summary="Delete conversation and unreferenced documents")
def remove_conversation(conv_id: str) -> Dict[str, Any]:
    """Deletes conversation, its messages, and removes unreferenced documents from disk."""
    result = delete_conversation(conv_id)
    if not result.get("deleted"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return result


def _process_confirmed_overwrite(confirmation_token: str, expected_target_file: Optional[str] = None) -> ChatResponse:
    from backend.storage import get_and_consume_overwrite_confirmation
    consumed, record, reason = get_and_consume_overwrite_confirmation(
        confirmation_token, target_file=expected_target_file
    )
    if not consumed:
        if reason == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Overwrite confirmation token not found.",
            )
        elif reason == "expired":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Overwrite confirmation token has expired (10m TTL exceeded).",
            )
        elif reason == "already_consumed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Overwrite confirmation token has already been consumed.",
            )
        elif reason == "path_mismatch":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Confirmation token is not valid for target file '{expected_target_file}'.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot confirm overwrite: {reason}",
            )

    target_rel = record["target_file"]
    # Revalidate path containment
    from backend.agent import validate_and_resolve_target_path
    val_p = validate_and_resolve_target_path(target_rel)
    canonical_rel = str(val_p.relative_to(WORKSPACE_DIR.resolve()))

    action_id = str(uuid.uuid4())
    steps = [
        {
            "step_number": 1,
            "action": "create_file",
            "target_file": canonical_rel,
            "content": record["file_content"],
            "description": f"Overwrite existing file '{canonical_rel}' inside workspace/ (explicitly confirmed by user).",
            "allow_overwrite": True,
        }
    ]

    store_action(
        action_id=action_id,
        conversation_id=record["conversation_id"],
        user_message=f"Confirmed overwrite for {canonical_rel}",
        steps=steps,
        mode=record["mode"],
        model_name=record.get("model_name"),
    )

    msg_text = f"Explicit overwrite confirmation received. Action plan created for '{canonical_rel}'. Human approval required before disk execution."
    add_message(
        conversation_id=record["conversation_id"],
        role="assistant",
        content=msg_text,
        metadata={
            "action_id": action_id,
            "status": "pending_approval",
            "intent": "create_file",
            "plan": steps,
            "file_preview": record["file_content"],
        },
    )

    return ChatResponse(
        action_id=action_id,
        conversation_id=record["conversation_id"],
        status="pending_approval",
        message=msg_text,
        ai_disclaimer="Explicit user confirmation verified. Action requires human approval before disk write.",
        supported=True,
        intent="create_file",
        mode=record["mode"],
        plan=[
            PlanStepSchema(
                step_number=1,
                action="create_file",
                target_file=canonical_rel,
                description=f"Overwrite existing file '{canonical_rel}' inside workspace/ (explicitly confirmed by user).",
                allow_overwrite=True,
            )
        ],
        file_preview=record["file_content"],
    )


@app.post(
    "/api/v1/actions/confirm-overwrite",
    response_model=ChatResponse,
    summary="Explicitly confirm overwriting an existing file using a server-issued token",
)
def confirm_overwrite(request: ConfirmOverwriteRequest) -> ChatResponse:
    """
    Consumes a server-issued overwrite confirmation token and creates an overwrite-authorized action.
    The confirmation token is bound strictly to the destination file and cannot be reused or forged.
    """
    return _process_confirmed_overwrite(request.confirmation_token, request.target_file)


@app.post("/api/v1/chat", response_model=ChatResponse, summary="Submit message and get a plan")
def chat(request: ChatRequest) -> ChatResponse:
    """
    Step 1 & 2: Understand and Plan
    When GEMINI_API_KEY is configured, generates plan using hosted Gemma 4.
    Otherwise, generates plan using the safe deterministic demo template.
    Does NOT execute any action yet.
    """
    if request.confirmation_token:
        return _process_confirmed_overwrite(request.confirmation_token)

    clean_message = request.message.strip()
    if not clean_message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Message cannot be empty or whitespace.",
        )

    # Ensure a conversation exists for local persistence
    conv_id = request.conversation_id or str(uuid.uuid4())
    add_message(
        conversation_id=conv_id,
        role="user",
        content=clean_message,
        metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
    )

    # --------------------------------------------------------------------------
    # 0. Direct Date & Time Handling (Zero Network, Reliable Runtime Timezone)
    # --------------------------------------------------------------------------
    if is_direct_datetime_query(clean_message):
        dt_answer = answer_datetime_query(clean_message)
        add_message(
            conversation_id=conv_id,
            role="assistant",
            content=dt_answer,
            metadata={
                "status": "unsupported",
                "intent": "general_question",
                "mode": "system_clock",
                "model": "datetime_runtime",
            },
        )
        return ChatResponse(
            action_id=None,
            conversation_id=conv_id,
            status="unsupported",
            message=dt_answer,
            ai_disclaimer="Answered directly from the backend system clock.",
            supported=False,
            intent="general_question",
            mode="system_clock",
            plan=[],
            file_preview=None,
        )

    # --------------------------------------------------------------------------
    # 0.5 Weather Handling (Open-Meteo Geocoding & Real-time Forecast)
    # --------------------------------------------------------------------------
    if is_weather_query(clean_message):
        weather_result = handle_weather_query(clean_message)
        w_status = weather_result.get("status", "success")
        w_msg = weather_result.get("message", "")
        mode_str = (
            "weather_service"
            if w_status == "success"
            else "weather_clarification"
            if w_status == "clarification_needed"
            else "weather_error"
        )
        disclaimer = "Retrieved via Open-Meteo meteorological and geocoding services."
        if weather_result.get("cached"):
            disclaimer += " (from recent 10-minute cache)"

        add_message(
            conversation_id=conv_id,
            role="assistant",
            content=w_msg,
            metadata={
                "status": "unsupported",
                "intent": "general_question",
                "mode": mode_str,
                "location": weather_result.get("location"),
            },
        )
        return ChatResponse(
            action_id=None,
            conversation_id=conv_id,
            status="unsupported",
            message=w_msg,
            ai_disclaimer=disclaimer,
            supported=False,
            intent="general_question",
            mode=mode_str,
            plan=[],
            file_preview=None,
        )

    # --------------------------------------------------------------------------
    # 1. Tavily Live Web Search (When current/external web information is requested)
    # --------------------------------------------------------------------------
    if should_search_web(clean_message):
        if not is_tavily_configured():
            msg = (
                "Web search is required for this query, but `TAVILY_API_KEY` is not configured "
                "on the backend. Please add your Tavily API key to the backend environment to enable live web search."
            )
            add_message(
                conversation_id=conv_id,
                role="assistant",
                content=msg,
                metadata={"status": "unsupported", "intent": "general_question", "mode": "search_unconfigured"},
            )
            return ChatResponse(
                action_id=None,
                conversation_id=conv_id,
                status="unsupported",
                message=msg,
                ai_disclaimer="Web search unavailable: TAVILY_API_KEY is not configured.",
                supported=False,
                intent="general_question",
                mode="search_unconfigured",
                plan=[],
                file_preview=None,
            )

        if not is_gemini_configured():
            msg = (
                "Web search retrieval succeeded, but synthesizing the results requires `GEMINI_API_KEY` "
                "to be configured on the backend. Demo mode does not support AI web synthesis."
            )
            add_message(
                conversation_id=conv_id,
                role="assistant",
                content=msg,
                metadata={"status": "unsupported", "intent": "general_question", "mode": "demo"},
            )
            return ChatResponse(
                action_id=None,
                conversation_id=conv_id,
                status="unsupported",
                message=msg,
                ai_disclaimer="Web synthesis requires GEMINI_API_KEY.",
                supported=False,
                intent="general_question",
                mode="demo",
                plan=[],
                file_preview=None,
            )

        try:
            search_data = execute_tavily_search(clean_query := clean_message)
            evidence_str, sources = format_search_evidence(search_data)
            context_slice = get_conversation_context_slice(conv_id, max_messages=6)
            if context_slice and context_slice[-1]["content"] == clean_message:
                context_slice = context_slice[:-1]

            synthesized_answer = synthesize_web_search(
                user_question=clean_message,
                search_evidence=evidence_str,
                context_slice=context_slice,
            )
        except SearchServiceError as exc:
            msg = f"Web search could not be completed: {exc}"
            add_message(
                conversation_id=conv_id,
                role="assistant",
                content=msg,
                metadata={"status": "unsupported", "intent": "general_question", "mode": "search_error"},
            )
            return ChatResponse(
                action_id=None,
                conversation_id=conv_id,
                status="unsupported",
                message=msg,
                ai_disclaimer="Web search query encountered an error.",
                supported=False,
                intent="general_question",
                mode="search_error",
                plan=[],
                file_preview=None,
            )
        except ModelClientError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to synthesize search results: {exc}",
                headers={"X-Conversation-Id": conv_id},
            )

        add_message(
            conversation_id=conv_id,
            role="assistant",
            content=synthesized_answer,
            metadata={
                "status": "unsupported",
                "intent": "general_question",
                "mode": "tavily_search",
                "sources": sources,
            },
        )
        return ChatResponse(
            action_id=None,
            conversation_id=conv_id,
            status="unsupported",
            message=synthesized_answer,
            ai_disclaimer="Grounded in real-time Tavily web search results synthesized by Gemma 4.",
            supported=False,
            intent="general_question",
            mode="tavily_search",
            plan=[],
            file_preview=None,
        )

    # --------------------------------------------------------------------------
    # Case A: Live Gemma 4 AI Planning
    # --------------------------------------------------------------------------
    if is_gemini_configured():
        model_name = get_model_name()
        # Retrieve bounded recent conversation context slice for follow-ups
        context_slice = get_conversation_context_slice(conv_id, max_messages=6)
        # Exclude the current message from slice so it isn't duplicated
        if context_slice and context_slice[-1]["content"] == clean_message:
            context_slice = context_slice[:-1]

        try:
            gemma_plan = generate_gemma_plan(clean_message, context_slice=context_slice)
        except ModelClientError as e:
            # Report provider/client error clearly without exposing credentials
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(e),
                headers={"X-Conversation-Id": conv_id},
            )

        # If Gemma determined the request is unsupported for file creation (e.g. general questions):
        if not gemma_plan.supported:
            add_message(
                conversation_id=conv_id,
                role="assistant",
                content=gemma_plan.assistant_message,
                metadata={
                    "status": "unsupported",
                    "intent": gemma_plan.intent,
                    "mode": "gemma",
                    "model": model_name,
                },
            )
            return ChatResponse(
                action_id=None,
                conversation_id=conv_id,
                status="unsupported",
                message=gemma_plan.assistant_message,
                ai_disclaimer=f"Gemma 4 ({model_name}) answered your request. No file action was needed.",
                supported=False,
                intent=gemma_plan.intent,
                mode="gemma",
                plan=[],
                file_preview=None,
            )

        # Supported request: Enforce strict backend safety boundaries:
        action_id = str(uuid.uuid4())
        # Use model-proposed target_file, or extract candidate from user message, or default to demo.txt
        target_filename = gemma_plan.target_file or "demo.txt"
        from backend.agent import validate_and_resolve_target_path
        try:
            target_path = validate_and_resolve_target_path(target_filename)
            rel_target = str(target_path.relative_to(WORKSPACE_DIR.resolve()))
        except (ValueError, PermissionError):
            rel_target = "demo.txt"

        # Check existing file: if destination already exists in workspace/, require explicit separate confirmation.
        # NEVER infer overwrite authorization from keywords in user message.
        dest_path = WORKSPACE_DIR / rel_target
        dest_exists = dest_path.exists()
        if dest_exists:
            import secrets
            from backend.storage import create_overwrite_confirmation
            token = secrets.token_urlsafe(24)
            create_overwrite_confirmation(
                token=token,
                conversation_id=conv_id,
                target_file=rel_target,
                file_content=gemma_plan.file_content,
                action_desc=gemma_plan.plan_description or f"Create '{rel_target}'",
                mode="gemma",
                model_name=model_name,
            )
            confirmation_msg = (
                f"File '{rel_target}' already exists in workspace/. "
                "To prevent accidental data loss, KAIROS requires explicit confirmation before creating an overwrite plan. "
                f"Please confirm if you want to overwrite '{rel_target}'."
            )
            add_message(
                conversation_id=conv_id,
                role="assistant",
                content=confirmation_msg,
                metadata={
                    "status": "confirmation_required",
                    "intent": "confirmation_required",
                    "mode": "gemma",
                    "target_file": rel_target,
                    "confirmation_token": token,
                },
            )
            return ChatResponse(
                action_id=None,
                conversation_id=conv_id,
                status="confirmation_required",
                message=confirmation_msg,
                ai_disclaimer=f"Gemma 4 ({model_name}): Target file '{rel_target}' already exists. Explicit confirmation is required to proceed with overwriting.",
                supported=False,
                intent="confirmation_required",
                mode="gemma",
                plan=[],
                file_preview=None,
                confirmation_token=token,
                confirmation_target_file=rel_target,
            )

        action_desc = gemma_plan.plan_description or f"Create '{rel_target}' inside workspace/ with the generated content."

        steps = [
            {
                "step_number": 1,
                "action": "create_file",
                "target_file": rel_target,
                "content": gemma_plan.file_content,
                "description": action_desc,
                "allow_overwrite": False,
            }
        ]

        # Store action durably in SQLite awaiting explicit human approval
        store_action(
            action_id=action_id,
            conversation_id=conv_id,
            user_message=clean_message,
            steps=steps,
            mode="gemma",
            model_name=model_name,
        )

        add_message(
            conversation_id=conv_id,
            role="assistant",
            content=gemma_plan.assistant_message,
            metadata={
                "action_id": action_id,
                "status": "pending_approval",
                "intent": gemma_plan.intent,
                "plan": steps,
                "file_preview": gemma_plan.file_content,
            },
        )

        return ChatResponse(
            action_id=action_id,
            conversation_id=conv_id,
            status="pending_approval",
            message=gemma_plan.assistant_message,
            ai_disclaimer=f"Powered by Gemma 4 ({model_name}). Action requires your explicit approval.",
            supported=True,
            intent=gemma_plan.intent,
            mode="gemma",
            plan=[
                PlanStepSchema(
                    step_number=s["step_number"],
                    action=s["action"],
                    target_file=s["target_file"],
                    description=s["description"],
                    allow_overwrite=s.get("allow_overwrite", False),
                )
                for s in steps
            ],
            file_preview=gemma_plan.file_content,
        )

    # --------------------------------------------------------------------------
    # Case B: Local Safe Demo Mode (When GEMINI_API_KEY is not configured)
    # --------------------------------------------------------------------------
    action_id = str(uuid.uuid4())
    # Extract requested filename if present, or fallback to demo.txt
    from backend.agent import validate_and_resolve_target_path
    import re
    demo_match = re.search(r"(?:into|to|file|named|name)\s+[`'\"]([a-zA-Z0-9_\-\./\\]+)[`'\"]", clean_message, re.IGNORECASE)
    if not demo_match:
        demo_match = re.search(r"[`'\"]([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)[`'\"]", clean_message)
    if not demo_match:
        demo_match = re.search(r"(?:into|to|file)\s+([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)", clean_message, re.IGNORECASE)

    target_filename = "demo.txt"
    if demo_match:
        cand = demo_match.group(1).replace("\\", "/").strip()
        try:
            val_p = validate_and_resolve_target_path(cand)
            target_filename = str(val_p.relative_to(WORKSPACE_DIR.resolve()))
        except (ValueError, PermissionError):
            target_filename = "demo.txt"

    dest_path = WORKSPACE_DIR / target_filename
    dest_exists = dest_path.exists()

    demo_content = (
        "========================================\n"
        "Hello from KAIROS Web API!\n"
        "Action approved via HTTP endpoint.\n"
        "Safely executed inside workspace/.\n"
        "========================================\n"
    )

    # If destination exists, require explicit separate confirmation.
    # NEVER infer overwrite authorization from keywords in user message.
    if dest_exists:
        import secrets
        from backend.storage import create_overwrite_confirmation
        token = secrets.token_urlsafe(24)
        create_overwrite_confirmation(
            token=token,
            conversation_id=conv_id,
            target_file=target_filename,
            file_content=demo_content,
            action_desc=f"Create a harmless text file '{target_filename}' inside workspace/.",
            mode="demo",
            model_name=None,
        )
        confirmation_msg = (
            f"File '{target_filename}' already exists in workspace/. "
            "To prevent accidental data loss, KAIROS requires explicit confirmation before creating an overwrite plan. "
            f"Please confirm if you want to overwrite '{target_filename}'."
        )
        add_message(
            conversation_id=conv_id,
            role="assistant",
            content=confirmation_msg,
            metadata={
                "status": "confirmation_required",
                "intent": "confirmation_required",
                "mode": "demo",
                "target_file": target_filename,
                "confirmation_token": token,
            },
        )
        return ChatResponse(
            action_id=None,
            conversation_id=conv_id,
            status="confirmation_required",
            message=confirmation_msg,
            ai_disclaimer=f"Demo mode: Target file '{target_filename}' already exists. Explicit confirmation is required to proceed with overwriting.",
            supported=False,
            intent="confirmation_required",
            mode="demo",
            plan=[],
            file_preview=None,
            confirmation_token=token,
            confirmation_target_file=target_filename,
        )

    desc_str = f"Create a harmless text file '{target_filename}' inside workspace/."

    steps = [
        {
            "step_number": 1,
            "action": "create_file",
            "target_file": target_filename,
            "content": demo_content,
            "description": desc_str,
            "allow_overwrite": False,
        }
    ]

    # Store action durably in SQLite awaiting explicit human approval
    store_action(
        action_id=action_id,
        conversation_id=conv_id,
        user_message=clean_message,
        steps=steps,
        mode="demo",
        model_name=None,
    )

    add_message(
        conversation_id=conv_id,
        role="assistant",
        content="Plan created. Explicit human approval is required before execution.",
        metadata={
            "action_id": action_id,
            "status": "pending_approval",
            "intent": "create_file",
            "plan": steps,
            "file_preview": demo_content,
        },
    )

    return ChatResponse(
        action_id=action_id,
        conversation_id=conv_id,
        status="pending_approval",
        message="Plan created. Explicit human approval is required before execution.",
        ai_disclaimer="Demo mode: Predefined safe template (no external AI model connected).",
        supported=True,
        intent="create_file",
        mode="demo",
        plan=[
            PlanStepSchema(
                step_number=s["step_number"],
                action=s["action"],
                target_file=s["target_file"],
                description=s["description"],
                allow_overwrite=s.get("allow_overwrite", False),
            )
            for s in steps
        ],
        file_preview=demo_content,
    )


@app.post("/api/v1/actions/{action_id}/approve", response_model=ApprovalResponse, summary="Approve an action")
def approve_action(action_id: str) -> ApprovalResponse:
    """
    Step 3, 4 & 5: Approval -> Act -> Verify
    Executes the pending plan if and only if:
      - The action exists in SQLite.
      - The action is currently in 'pending_approval' status and unexpired.
      - Claimed atomically so concurrent or duplicate approvals execute at most once.
      - Action type is strictly 'create_file' and target resolves within workspace/.
    """
    # 1. Attempt atomic claim in SQLite
    claimed, action_record, reason = claim_action_for_execution(action_id)

    if not claimed:
        if reason == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Action ID '{action_id}' not found.",
            )
        elif reason == "expired":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Action '{action_id}' cannot be approved because it has expired (24h TTL exceeded).",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Action '{action_id}' cannot be approved because its status is already '{reason}'.",
            )

    results: List[ActionResult] = []

    try:
        # 2. Execute approved steps safely inside workspace/
        for step in action_record["steps"]:
            if step["action"] == "create_file":
                # Strict security check: write atomically and revalidate workspace containment
                safe_target_path = safe_atomic_write(
                    step["target_file"],
                    step["content"],
                    allow_overwrite=step.get("allow_overwrite", False),
                )

                # Verification: ensure file exists and is not empty
                if not safe_target_path.exists() or safe_target_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"Verification failed: {safe_target_path.name} was not properly written."
                    )


                rel_written = str(safe_target_path.relative_to(WORKSPACE_DIR.resolve()))
                results.append(
                    ActionResult(
                        step_number=step["step_number"],
                        target_file=f"workspace/{rel_written}",
                        status="completed",
                        bytes_written=safe_target_path.stat().st_size,
                    )
                )
            else:
                raise ValueError(f"Unsupported action type: {step['action']}")

        # 3. Mark as completed in SQLite
        complete_action(action_id)

    except Exception as exc:
        # Mark as failed in SQLite so it can be investigated or retried
        fail_action(action_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR if isinstance(exc, RuntimeError) else status.HTTP_400_BAD_REQUEST,
            detail=f"Execution error: {exc}",
        )

    return ApprovalResponse(
        action_id=action_id,
        status="completed",
        message="Action approved, executed, and verified successfully.",
        results=results,
    )


@app.post("/api/v1/actions/{action_id}/reject", response_model=RejectionResponse, summary="Reject an action")
def reject_action(action_id: str) -> RejectionResponse:
    """
    Rejection Gate:
    Marks the pending action as rejected in SQLite.
    Guarantees no files are created, modified, or executed.
    """
    success, reason = reject_action_record(action_id)

    if not success:
        if reason == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Action ID '{action_id}' not found.",
            )
        elif reason == "expired":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Action '{action_id}' cannot be rejected because it has expired.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Action '{action_id}' cannot be rejected because its status is already '{reason}'.",
            )

    return RejectionResponse(
        action_id=action_id,
        status="rejected",
        message="Action was rejected by user. No files were created or modified.",
    )


@app.get("/api/v1/actions/{action_id}", summary="Get action status")
def get_action_status(action_id: str) -> Dict[str, Any]:
    """Inspect the current state of any action by ID from SQLite."""
    action = get_action(action_id)
    if not action:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Action ID '{action_id}' not found.",
        )
    return action


# ==============================================================================
# 5. PDF Intelligence Endpoints (Read-Only)
# ==============================================================================

class PdfUploadResponse(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    size_bytes: int
    status: str
    is_scanned: bool = False
    ocr_available: bool = True
    note: str
    file_type: str = "pdf"


class UrlIngestRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2000, description="Public web URL to safely fetch and ingest.")
    conversation_id: Optional[str] = Field(default=None, description="Optional active conversation ID to link the web document to.")


class PdfAnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question or instruction about the PDF or image.")
    conversation_id: Optional[str] = Field(default=None, description="Optional active conversation ID to record Q&A.")


class PdfAnalysisResponse(BaseModel):
    doc_id: str
    filename: str
    question: str
    answer: str
    truncated: bool
    pages_included: int
    total_pages: int
    is_scanned: bool = False
    is_mixed: bool = False
    is_image: bool = False
    is_url: bool = False
    native_pages_count: int = 0
    ocr_pages_count: int = 0
    ocr_available: bool = True
    mode: str = "gemma"
    conversation_id: Optional[str] = None


@app.post(
    "/api/v1/pdf/upload",
    response_model=PdfUploadResponse,
    summary="Upload and validate a PDF for read-only Q&A",
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdf(file: UploadFile = File(...), conversation_id: Optional[str] = None) -> PdfUploadResponse:
    """
    Accepts multipart PDF upload.
    Validates PDF magic bytes, max 10 MB size, max 50 pages, encryption status.
    Stores safe file inside workspace/pdfs/<server-uuid>.pdf and records in SQLite.
    """
    chunk_limit = MAX_PDF_BYTES + 1
    file_bytes = await file.read(chunk_limit)

    if len(file_bytes) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds the maximum allowed size of {MAX_PDF_BYTES // (1024 * 1024)} MB.",
        )

    original_name = file.filename or "unnamed.pdf"

    try:
        metadata = validate_and_store_pdf(file_bytes, original_name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )

    # Persist document record to SQLite database
    store_document_record(
        doc_id=metadata["doc_id"],
        filename=metadata["filename"],
        storage_path=metadata["storage_path"],
        page_count=metadata["page_count"],
        size_bytes=metadata["size_bytes"],
        sha256_prefix=metadata["sha256_prefix"],
        ocr_available=metadata.get("ocr_available", True),
    )

    if conversation_id:
        link_document_to_conversation(conversation_id, metadata["doc_id"])

    return PdfUploadResponse(
        doc_id=metadata["doc_id"],
        filename=metadata["filename"],
        page_count=metadata["page_count"],
        size_bytes=metadata["size_bytes"],
        status="ready",
        is_scanned=False,
        ocr_available=metadata.get("ocr_available", True),
        note="Document stored locally and registered for analysis.",
        file_type="pdf",
    )


@app.post(
    "/api/v1/image/upload",
    response_model=PdfUploadResponse,
    summary="Upload and validate an image (PNG/JPEG/WebP) for read-only OCR Q&A",
    status_code=status.HTTP_201_CREATED,
)
async def upload_image(file: UploadFile = File(...), conversation_id: Optional[str] = None) -> PdfUploadResponse:
    """
    Accepts multipart image upload (PNG, JPEG, WebP).
    Validates image signatures (magic bytes), enforces strict 5 MB limit.
    Performs decompression bomb and corruption verification via Pillow.
    Stores safe file inside workspace/pdfs/<server-uuid>.<ext> and records in SQLite.
    """
    chunk_limit = MAX_IMAGE_BYTES + 1
    file_bytes = await file.read(chunk_limit)

    if len(file_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Image exceeds the maximum allowed size of {MAX_IMAGE_BYTES // (1024 * 1024)} MB ({len(file_bytes)} bytes uploaded).",
        )

    original_name = file.filename or "unnamed.png"
    detected_type = detect_file_type(file_bytes)
    if detected_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Unsupported or unrecognized image format. KAIROS accepts PNG, JPEG, and WebP images.",
        )

    try:
        metadata = validate_and_store_image(file_bytes, original_name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )

    store_document_record(
        doc_id=metadata["doc_id"],
        filename=metadata["filename"],
        storage_path=metadata["storage_path"],
        page_count=metadata["page_count"],
        size_bytes=metadata["size_bytes"],
        sha256_prefix=metadata["sha256_prefix"],
        ocr_available=metadata.get("ocr_available", True),
    )

    if conversation_id:
        link_document_to_conversation(conversation_id, metadata["doc_id"])

    return PdfUploadResponse(
        doc_id=metadata["doc_id"],
        filename=metadata["filename"],
        page_count=metadata["page_count"],
        size_bytes=metadata["size_bytes"],
        status="ready",
        is_scanned=True,
        ocr_available=metadata.get("ocr_available", True),
        note="Image stored locally and registered for OCR analysis.",
        file_type="image",
    )


@app.post(
    "/api/v1/attachment/upload",
    response_model=PdfUploadResponse,
    summary="Upload and validate an attachment (PDF or image) for read-only Q&A",
    status_code=status.HTTP_201_CREATED,
)
async def upload_attachment(file: UploadFile = File(...), conversation_id: Optional[str] = None) -> PdfUploadResponse:
    """
    Accepts multipart upload for PDF documents and PNG/JPEG/WebP images.
    Validates file signatures (magic bytes), enforces strict size limits (10 MB for PDFs, 5 MB for images).
    Stores safe file inside workspace/pdfs/<server-uuid>.<ext> and records in SQLite.
    """
    chunk_limit = MAX_PDF_BYTES + 1
    file_bytes = await file.read(chunk_limit)

    if len(file_bytes) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds the maximum allowed size of {MAX_PDF_BYTES // (1024 * 1024)} MB.",
        )

    original_name = file.filename or "unnamed"
    file_type_detected = detect_file_type(file_bytes)

    if file_type_detected == "application/pdf":
        try:
            metadata = validate_and_store_pdf(file_bytes, original_name)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            )
        note = "Document stored locally and registered for analysis."
        is_scanned = False
        file_kind = "pdf"
    elif file_type_detected in SUPPORTED_IMAGE_TYPES:
        if len(file_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Image exceeds the maximum allowed size of {MAX_IMAGE_BYTES // (1024 * 1024)} MB ({len(file_bytes)} bytes uploaded).",
            )
        try:
            metadata = validate_and_store_image(file_bytes, original_name)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            )
        note = "Image stored locally and registered for OCR analysis."
        is_scanned = True
        file_kind = "image"
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Unsupported file format. KAIROS accepts PDF documents and PNG, JPEG, or WebP images.",
        )

    store_document_record(
        doc_id=metadata["doc_id"],
        filename=metadata["filename"],
        storage_path=metadata["storage_path"],
        page_count=metadata["page_count"],
        size_bytes=metadata["size_bytes"],
        sha256_prefix=metadata["sha256_prefix"],
        ocr_available=metadata.get("ocr_available", True),
    )

    if conversation_id:
        link_document_to_conversation(conversation_id, metadata["doc_id"])

    return PdfUploadResponse(
        doc_id=metadata["doc_id"],
        filename=metadata["filename"],
        page_count=metadata["page_count"],
        size_bytes=metadata["size_bytes"],
        status="ready",
        is_scanned=is_scanned,
        ocr_available=metadata.get("ocr_available", True),
        note=note,
        file_type=file_kind,
    )


@app.post(
    "/api/v1/url/ingest",
    response_model=PdfUploadResponse,
    summary="Safely fetch and ingest a public webpage for read-only Q&A",
    status_code=status.HTTP_201_CREATED,
)
def ingest_url(request: UrlIngestRequest) -> PdfUploadResponse:
    """
    Safely fetches a webpage with connection pinning and SSRF protection.
    Extracts title and readable text (up to 8,000 chars), stores as a text document,
    and registers it for read-only Gemma Q&A.
    """
    raw_url = request.url.strip()
    if not raw_url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="URL cannot be empty or whitespace.",
        )

    try:
        final_url, response_text, content_type = safe_fetch_url(raw_url)
    except UrlValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )
    except SsrfBlockedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Security Alert: Requested URL or resolved IP is forbidden: {exc}",
        )
    except UrlContentTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        )
    except UrlUnsupportedContentTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )
    except UrlFetchError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch webpage: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unexpected error while fetching URL: {exc}",
        )

    # Extract clean text and title
    extracted = extract_html_content(response_text, final_url)
    page_text = extracted["text"].strip()
    if not page_text:
        page_text = f"[Webpage '{extracted['title']}' loaded, but contained no readable text.]"

    doc_id = str(uuid.uuid4())
    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    storage_file = PDF_STORAGE_DIR / f"{doc_id}.txt"
    storage_file.write_text(page_text, encoding="utf-8")

    import hashlib
    content_bytes = page_text.encode("utf-8")
    sha256_prefix = hashlib.sha256(content_bytes).hexdigest()[:16]
    display_title = extracted["title"] or final_url

    meta = {
        "doc_id": doc_id,
        "filename": display_title,
        "page_count": 1,
        "size_bytes": len(content_bytes),
        "sha256_prefix": sha256_prefix,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "storage_path": str(storage_file),
        "file_type": "url",
        "ocr_available": True,
    }
    pdf_registry[doc_id] = meta

    store_document_record(
        doc_id=doc_id,
        filename=display_title,
        storage_path=str(storage_file),
        page_count=1,
        size_bytes=len(content_bytes),
        sha256_prefix=sha256_prefix,
        ocr_available=True,
    )

    if request.conversation_id:
        link_document_to_conversation(request.conversation_id, doc_id)

    return PdfUploadResponse(
        doc_id=doc_id,
        filename=display_title,
        page_count=1,
        size_bytes=len(content_bytes),
        status="ready",
        is_scanned=False,
        ocr_available=True,
        note=f"Webpage content extracted ({len(page_text)} chars) and ready for analysis.",
        file_type="url",
    )


@app.post(
    "/api/v1/pdf/{doc_id}/analyze",
    response_model=PdfAnalysisResponse,
    summary="Ask a question about an uploaded PDF (read-only)",
)
def analyze_pdf(doc_id: str, request: PdfAnalyzeRequest) -> PdfAnalysisResponse:
    """
    Read-only PDF Q&A endpoint.
    Sends bounded document context and user question to hosted Gemma AI.
    Never executes commands, creates files, or alters the workspace.
    """
    # 1. Validate doc_id format (must be valid UUID4 to prevent traversal attacks)
    try:
        uuid.UUID(doc_id, version=4)
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    # 2. Check metadata registry
    metadata = get_doc_metadata(doc_id)
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found. It may have expired or backend was restarted.",
        )

    # 3. Check model configuration
    if not is_gemini_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document and image analysis requires GEMINI_API_KEY to be configured. Demo mode does not support document or image Q&A.",
        )

    # 4. Prepare bounded context
    try:
        context_data = prepare_model_context(doc_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # 5. Call Gemma model client with retry protection
    conv_id = request.conversation_id or str(uuid.uuid4())
    try:
        answer = analyze_pdf_content(
            document_context=context_data["context"],
            user_question=request.question,
            truncated=context_data["truncated"],
        )
    except ModelClientError as exc:
        link_document_to_conversation(conv_id, doc_id)
        add_message(
            conversation_id=conv_id,
            role="user",
            content=request.question.strip(),
            metadata={"doc_id": doc_id, "filename": metadata["filename"]},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
            headers={"X-Conversation-Id": conv_id},
        )

    # 6. Persist question & answer to SQLite
    link_document_to_conversation(conv_id, doc_id)
    add_message(
        conversation_id=conv_id,
        role="user",
        content=request.question.strip(),
        metadata={"doc_id": doc_id, "filename": metadata["filename"]},
    )
    add_message(
        conversation_id=conv_id,
        role="assistant",
        content=answer,
        metadata={
            "doc_id": doc_id,
            "filename": metadata["filename"],
            "is_scanned": context_data.get("is_scanned", False),
            "is_mixed": context_data.get("is_mixed", False),
            "is_image": context_data.get("is_image", False),
            "is_url": context_data.get("is_url", False),
            "native_pages_count": context_data.get("native_pages_count", 0),
            "ocr_pages_count": context_data.get("ocr_pages_count", 0),
            "ocr_available": context_data.get("ocr_available", True),
            "status": "analysis_complete",
        },
    )

    return PdfAnalysisResponse(
        doc_id=doc_id,
        filename=metadata["filename"],
        question=request.question.strip(),
        answer=answer,
        truncated=context_data["truncated"],
        pages_included=context_data["pages_included"],
        total_pages=context_data["total_pages"],
        is_scanned=context_data.get("is_scanned", False),
        is_mixed=context_data.get("is_mixed", False),
        is_image=context_data.get("is_image", False),
        is_url=context_data.get("is_url", False),
        native_pages_count=context_data.get("native_pages_count", 0),
        ocr_pages_count=context_data.get("ocr_pages_count", 0),
        ocr_available=context_data.get("ocr_available", True),
        mode="gemma",
        conversation_id=conv_id,
    )
