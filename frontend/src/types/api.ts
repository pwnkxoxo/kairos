export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  mode?: string;
  model?: string;
}

export interface PlanStep {
  step_number: number;
  action: string;
  target_file: string;
  description: string;
  allow_overwrite?: boolean;
}

export interface ChatResponse {
  action_id?: string | null;
  conversation_id?: string | null;
  status: string; // 'pending_approval' | 'unsupported'
  message: string;
  ai_disclaimer: string;
  supported: boolean;
  intent?: string;
  mode: string; // 'gemma' | 'demo'
  plan: PlanStep[];
  file_preview?: string | null;
  confirmation_token?: string | null;
  confirmation_target_file?: string | null;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface MessageItem {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata?: Record<string, any>;
  created_at: string;
}

export interface AttachedDocInfo {
  doc_id: string;
  filename: string;
  storage_path: string;
  page_count: number;
  size_bytes: number;
  ocr_available: boolean;
}

export interface ConversationDetail {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: MessageItem[];
  documents: AttachedDocInfo[];
}

export interface ActionResult {
  step_number: number;
  target_file: string;
  status: string;
  bytes_written: number;
}

export interface ApprovalResponse {
  action_id: string;
  status: string; // 'completed'
  message: string;
  results: ActionResult[];
}

export interface RejectionResponse {
  action_id: string;
  status: string; // 'rejected'
  message: string;
}

export type ActionStatus =
  | 'idle'
  | 'planning'
  | 'pending_approval'
  | 'unsupported'
  | 'executing'
  | 'completed'
  | 'rejected'
  | 'error'
  | 'analyzing'
  | 'analysis_complete';

// Document & Image Intelligence types
export interface PdfUploadResponse {
  doc_id: string;
  filename: string;
  page_count: number;
  size_bytes: number;
  status: string;
  is_scanned: boolean;
  ocr_available?: boolean;
  note: string;
  file_type?: string;
}

export interface PdfAnalysisResponse {
  doc_id: string;
  filename: string;
  question: string;
  answer: string;
  truncated: boolean;
  pages_included: number;
  total_pages: number;
  is_scanned: boolean;
  is_mixed?: boolean;
  is_image?: boolean;
  is_url?: boolean;
  native_pages_count?: number;
  ocr_pages_count?: number;
  ocr_available?: boolean;
  mode: string;
  conversation_id?: string;
}

export type AttachmentProcessingState =
  | 'pending'
  | 'uploading'
  | 'extracting_text'
  | 'reading_ocr'
  | 'analyzing'
  | 'ready'
  | 'error';

export interface UrlIngestRequest {
  url: string;
  conversation_id?: string;
}

export type UrlIngestResponse = PdfUploadResponse;

export interface AttachmentItem {
  id: string;
  file?: File;
  filename: string;
  sizeBytes: number;
  uploadState: AttachmentProcessingState;
  docId?: string;
  pageCount?: number;
  isScanned?: boolean;
  isMixed?: boolean;
  isImage?: boolean;
  isUrl?: boolean;
  fileType?: 'pdf' | 'image' | 'url';
  ocrAvailable?: boolean;
  errorMessage?: string;
}
