import {
  HealthResponse,
  ChatResponse,
  ApprovalResponse,
  RejectionResponse,
  PdfUploadResponse,
  PdfAnalysisResponse,
  ConversationSummary,
  ConversationDetail,
} from '../types/api';

export class ApiError extends Error {
  statusCode?: number;
  conversationId?: string;

  constructor(message: string, statusCode?: number, conversationId?: string) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.conversationId = conversationId;
  }
}

const DEFAULT_TIMEOUT_MS = 25000;

async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      ...options,
      signal: options.signal || controller.signal,
    });
    return res;
  } catch (err: any) {
    if (err.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs / 1000} seconds. Please check backend network connectivity.`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Checks if the FastAPI backend service is reachable and healthy.
 */
export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetchWithTimeout('/health', {}, 8000);
  if (!response.ok) {
    throw new ApiError(`Health check failed with status: ${response.status}`, response.status);
  }
  return response.json();
}

/**
 * Retrieves all stored conversations with optional search query.
 */
export async function fetchConversations(query?: string): Promise<ConversationSummary[]> {
  const url = query ? `/api/v1/conversations?q=${encodeURIComponent(query)}` : '/api/v1/conversations';
  const response = await fetchWithTimeout(url, {}, 10000);
  if (!response.ok) {
    throw new ApiError(`Failed to load conversations: ${response.status}`, response.status);
  }
  return response.json();
}

/**
 * Creates a new conversation.
 */
export async function createConversation(title?: string): Promise<ConversationSummary> {
  const response = await fetchWithTimeout('/api/v1/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: title || 'New Chat' }),
  }, 10000);
  if (!response.ok) {
    throw new ApiError(`Failed to create conversation: ${response.status}`, response.status);
  }
  return response.json();
}

/**
 * Loads an existing conversation with full message history and attached documents.
 */
export async function fetchConversation(convId: string): Promise<ConversationDetail> {
  const response = await fetchWithTimeout(`/api/v1/conversations/${convId}`, {}, 10000);
  if (!response.ok) {
    throw new ApiError(`Failed to load conversation: ${response.status}`, response.status);
  }
  return response.json();
}

/**
 * Renames an existing conversation title.
 */
export async function renameConversation(convId: string, title: string): Promise<{ id: string; title: string }> {
  const response = await fetchWithTimeout(`/api/v1/conversations/${convId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  }, 10000);
  if (!response.ok) {
    throw new ApiError(`Failed to rename conversation: ${response.status}`, response.status);
  }
  return response.json();
}

/**
 * Deletes a conversation and any unreferenced documents.
 */
export async function deleteConversation(convId: string): Promise<{ conversation_id: string; deleted: boolean }> {
  const response = await fetchWithTimeout(`/api/v1/conversations/${convId}`, {
    method: 'DELETE',
  }, 15000);
  if (!response.ok) {
    throw new ApiError(`Failed to delete conversation: ${response.status}`, response.status);
  }
  return response.json();
}

/**
 * Submits a user prompt to KAIROS to generate a plan awaiting approval.
 */
export async function sendChatMessage(message: string, conversationId?: string): Promise<ChatResponse> {
  let response: Response;
  try {
    response = await fetchWithTimeout('/api/v1/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message, conversation_id: conversationId }),
    }, 75000);
  } catch (err: any) {
    throw new ApiError(err.message || 'Chat request failed.', undefined, conversationId);
  }

  const serverConvId = response.headers.get('X-Conversation-Id') || conversationId;

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new ApiError(
      errorData?.detail || `Chat request failed with status: ${response.status}`,
      response.status,
      serverConvId || undefined
    );
  }

  return response.json();
}

/**
 * Confirms overwriting an existing file using a server-issued confirmation token.
 */
export async function confirmOverwrite(confirmationToken: string, targetFile?: string): Promise<ChatResponse> {
  let response: Response;
  try {
    response = await fetchWithTimeout('/api/v1/actions/confirm-overwrite', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirmation_token: confirmationToken, target_file: targetFile }),
    }, 15000);
  } catch (err: any) {
    throw new ApiError(err.message || 'Overwrite confirmation failed.');
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new ApiError(
      errorData?.detail || `Overwrite confirmation failed with status: ${response.status}`,
      response.status
    );
  }

  return response.json();
}

/**
 * Approves a pending action, triggering safe execution in workspace/ and verification.
 */
export async function approveAction(actionId: string): Promise<ApprovalResponse> {
  const response = await fetchWithTimeout(`/api/v1/actions/${actionId}/approve`, {
    method: 'POST',
  }, 15000);

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new ApiError(
      errorData?.detail || `Approval failed with status: ${response.status}`,
      response.status
    );
  }

  return response.json();
}

/**
 * Rejects a pending action, guaranteeing no files are written.
 */
export async function rejectAction(actionId: string): Promise<RejectionResponse> {
  const response = await fetchWithTimeout(`/api/v1/actions/${actionId}/reject`, {
    method: 'POST',
  }, 10000);

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new ApiError(
      errorData?.detail || `Rejection failed with status: ${response.status}`,
      response.status
    );
  }

  return response.json();
}

/**
 * Uploads and validates a PDF document or image for read-only analysis.
 */
export async function uploadPdf(file: File, conversationId?: string): Promise<PdfUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const url = conversationId
    ? `/api/v1/attachment/upload?conversation_id=${encodeURIComponent(conversationId)}`
    : '/api/v1/attachment/upload';

  const response = await fetchWithTimeout(url, {
    method: 'POST',
    body: formData,
  }, 25000);

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new ApiError(
      errorData?.detail || `Upload failed with status: ${response.status}`,
      response.status
    );
  }

  return response.json();
}

export const uploadAttachment = uploadPdf;

/**
 * Safely fetches and ingests a public webpage for read-only Q&A.
 */
export async function ingestUrl(url: string, conversationId?: string): Promise<PdfUploadResponse> {
  const response = await fetchWithTimeout('/api/v1/url/ingest', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ url, conversation_id: conversationId }),
  }, 25000);

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new ApiError(
      errorData?.detail || `Webpage ingest failed with status: ${response.status}`,
      response.status
    );
  }

  return response.json();
}

/**
 * Asks a question about an uploaded PDF document.
 * Returns the model's answer.
 */
export async function analyzePdf(
  docId: string,
  question: string,
  conversationId?: string
): Promise<PdfAnalysisResponse> {
  let response: Response;
  try {
    response = await fetchWithTimeout(`/api/v1/pdf/${docId}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question, conversation_id: conversationId }),
    }, 75000);
  } catch (err: any) {
    throw new ApiError(err.message || 'Document analysis request failed.', undefined, conversationId);
  }

  const serverConvId = response.headers.get('X-Conversation-Id') || conversationId;

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new ApiError(
      errorData?.detail || `PDF analysis failed with status: ${response.status}`,
      response.status,
      serverConvId || undefined
    );
  }

  return response.json();
}
