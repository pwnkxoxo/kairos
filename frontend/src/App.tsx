import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  HealthResponse,
  ChatResponse,
  ApprovalResponse,
  RejectionResponse,
  PdfAnalysisResponse,
  AttachmentItem,
  ActionStatus,
  ConversationSummary,
  MessageItem,
} from './types/api';
import { Sidebar } from './components/Sidebar';
import { Timeline } from './components/Timeline';
import { PlanCard } from './components/PlanCard';
import { ExecutionResult } from './components/ExecutionResult';
import { ChatInput } from './components/ChatInput';
import { MarkdownRenderer } from './components/MarkdownRenderer';
import {
  fetchHealth,
  sendChatMessage,
  approveAction,
  rejectAction,
  fetchConversations,
  createConversation,
  fetchConversation,
  renameConversation,
  deleteConversation,
  uploadPdf,
  analyzePdf,
  ingestUrl,
  confirmOverwrite,
} from './services/api';
import {
  ShieldCheck,
  User,
  Sparkles,
  FileText,
  Image as ImageIcon,
  Globe,
  Copy,
  Check,
  Volume2,
  VolumeX,
  AlertCircle,
  AlertTriangle,
  RotateCcw,
  Loader2,
  X,
} from 'lucide-react';

export const App: React.FC = () => {
  // Backend Health
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);

  // Conversations & Local History
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Workflow State for active turn
  const [status, setStatus] = useState<ActionStatus>('idle');
  const [lastPrompt, setLastPrompt] = useState<string>('');

  // Active Pending Action / Results
  const [planData, setPlanData] = useState<ChatResponse | null>(null);
  const [approvalResult, setApprovalResult] = useState<ApprovalResponse | null>(null);
  const [rejectionResult, setRejectionResult] = useState<RejectionResponse | null>(null);
  const [pdfAnalysisResult, setPdfAnalysisResult] = useState<PdfAnalysisResponse | null>(null);

  // Button load states
  const [activeButton, setActiveButton] = useState<'approve' | 'reject' | 'confirm_overwrite' | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Attachments
  const [attachments, setAttachments] = useState<AttachmentItem[]>([]);

  // Speech & Copy feedback states
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [speakingId, setSpeakingId] = useState<string | null>(null);

  // Chat scroll container
  const chatScrollRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, planData, approvalResult, rejectionResult, pdfAnalysisResult, status, errorMessage]);

  // Read aloud toggle using SpeechSynthesis
  const handleToggleSpeak = (text: string, id: string) => {
    if (!('speechSynthesis' in window)) {
      setErrorMessage('Speech synthesis is not supported by your browser.');
      return;
    }

    if (speakingId === id) {
      window.speechSynthesis.cancel();
      setSpeakingId(null);
      return;
    }

    window.speechSynthesis.cancel();

    // Strip markdown formatting for cleaner speech synthesis
    const cleanSpeechText = text
      .replace(/```[\s\S]*?```/g, 'Code block omitted.')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/\$[^$]+\$/g, 'Mathematical expression.')
      .replace(/[#*_~\[\]]/g, ' ')
      .trim();

    const utterance = new SpeechSynthesisUtterance(cleanSpeechText);
    utterance.lang = 'en-IN';
    utterance.rate = 1.0;

    utterance.onend = () => {
      setSpeakingId(null);
    };

    utterance.onerror = () => {
      setSpeakingId(null);
    };

    setSpeakingId(id);
    window.speechSynthesis.speak(utterance);
  };

  // Copy answer to clipboard
  const handleCopyText = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  // Health Check
  const checkBackendHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  // Load conversation list from local SQLite
  const loadConversations = useCallback(async (selectId?: string) => {
    try {
      const list = await fetchConversations();
      setConversations(list);
      if (selectId) {
        setActiveConversationId(selectId);
      }
      return list;
    } catch (err) {
      console.error('Error loading conversations:', err);
      return [];
    }
  }, []);

  // Select an existing conversation from sidebar
  const handleSelectConversation = async (id: string) => {
    if (id === activeConversationId && messages.length > 0) return;
    setActiveConversationId(id);
    setPlanData(null);
    setApprovalResult(null);
    setRejectionResult(null);
    setPdfAnalysisResult(null);
    setErrorMessage(null);
    setStatus('idle');

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      setSpeakingId(null);
    }

    setHistoryLoading(true);
    try {
      const detail = await fetchConversation(id);
      setMessages(detail.messages || []);
      if (detail.documents && detail.documents.length > 0) {
        const doc = detail.documents[detail.documents.length - 1];
        const ext = doc.filename.toLowerCase().split('.').pop() || '';
        const isImg = ['png', 'jpg', 'jpeg', 'webp'].includes(ext);
        const isUrl = doc.filename.startsWith('http://') || doc.filename.startsWith('https://') || ext === 'txt';
        const fileKind = isUrl ? 'url' : (isImg ? 'image' : 'pdf');
        setAttachments([
          {
            id: doc.doc_id,
            file: isUrl ? undefined : new File([], doc.filename, { type: isImg ? 'image/png' : 'application/pdf' }),
            filename: doc.filename,
            sizeBytes: doc.size_bytes,
            uploadState: 'ready',
            docId: doc.doc_id,
            pageCount: doc.page_count,
            ocrAvailable: doc.ocr_available,
            fileType: fileKind,
            isImage: isImg,
            isUrl: isUrl,
          },
        ]);
      } else {
        setAttachments([]);
      }
    } catch (err) {
      console.error('Failed to load conversation messages:', err);
      setErrorMessage('Failed to load messages for this conversation.');
    } finally {
      setHistoryLoading(false);
    }
  };

  // Initial load: fetch health, list conversations, and restore the latest conversation
  useEffect(() => {
    checkBackendHealth();
    loadConversations().then((list) => {
      if (list && list.length > 0 && !activeConversationId) {
        handleSelectConversation(list[0].id);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Start a fresh, empty conversation
  const handleNewChat = async () => {
    try {
      const newConv = await createConversation('New Chat');
      await loadConversations(newConv.id);
      setActiveConversationId(newConv.id);
      setMessages([]);
      setStatus('idle');
      setPlanData(null);
      setApprovalResult(null);
      setRejectionResult(null);
      setPdfAnalysisResult(null);
      setAttachments([]);
      setErrorMessage(null);
      setLastPrompt('');
    } catch (err) {
      console.error('Failed to create new conversation:', err);
    }
  };

  // Rename a conversation
  const handleRenameConversation = async (id: string, newTitle: string) => {
    try {
      await renameConversation(id, newTitle);
      await loadConversations(activeConversationId || undefined);
    } catch (err) {
      setErrorMessage('Failed to rename conversation.');
    }
  };

  // Delete a conversation
  const handleDeleteConversation = async (id: string) => {
    try {
      await deleteConversation(id);
      const remaining = conversations.filter((c) => c.id !== id);
      setConversations(remaining);
      if (activeConversationId === id) {
        if (remaining.length > 0) {
          handleSelectConversation(remaining[0].id);
        } else {
          setActiveConversationId(null);
          setMessages([]);
          setStatus('idle');
        }
      }
    } catch (err) {
      setErrorMessage('Failed to delete conversation.');
    }
  };

  // Attach PDF or Image
  const handleAddAttachment = async (file: File) => {
    const ext = file.name.toLowerCase().split('.').pop() || '';
    const isPdf = ext === 'pdf' || file.type === 'application/pdf';
    const isImage =
      ['png', 'jpg', 'jpeg', 'webp'].includes(ext) ||
      ['image/png', 'image/jpeg', 'image/webp'].includes(file.type);

    if (!isPdf && !isImage) {
      setErrorMessage(
        'Unsupported file format. KAIROS accepts PDF documents (up to 10 MB) and PNG, JPEG, or WebP images (up to 5 MB).'
      );
      return;
    }

    const maxBytes = isPdf ? 10 * 1024 * 1024 : 5 * 1024 * 1024;
    const maxMb = isPdf ? 10 : 5;
    if (file.size > maxBytes) {
      setErrorMessage(
        `${isPdf ? 'PDF' : 'Image'} exceeds ${maxMb} MB limit (${(file.size / 1024 / 1024).toFixed(1)} MB).`
      );
      return;
    }

    const attachmentId = Math.random().toString(36).substring(2, 9);
    const newAttachment: AttachmentItem = {
      id: attachmentId,
      file,
      filename: file.name,
      sizeBytes: file.size,
      uploadState: 'uploading',
      fileType: isImage ? 'image' : 'pdf',
      isImage,
    };

    setAttachments([newAttachment]);
    setErrorMessage(null);

    try {
      const uploadResp = await uploadPdf(file, activeConversationId || undefined);
      const isImg = uploadResp.file_type === 'image' || isImage;
      setAttachments((prev) =>
        prev.map((att) =>
          att.id === attachmentId
            ? {
                ...att,
                uploadState: 'ready',
                docId: uploadResp.doc_id,
                pageCount: uploadResp.page_count,
                isScanned: uploadResp.is_scanned,
                ocrAvailable: uploadResp.ocr_available,
                fileType: isImg ? 'image' : 'pdf',
                isImage: isImg,
              }
            : att
        )
      );
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Upload failed';
      setAttachments((prev) =>
        prev.map((att) =>
          att.id === attachmentId
            ? {
                ...att,
                uploadState: 'error',
                errorMessage: errMsg,
              }
            : att
        )
      );
      setErrorMessage(errMsg);
    }
  };

  const handleAddUrl = async (url: string) => {
    const attachmentId = Math.random().toString(36).substring(2, 9);

    const newAttachment: AttachmentItem = {
      id: attachmentId,
      filename: url,
      sizeBytes: 0,
      uploadState: 'uploading',
      fileType: 'url',
      isUrl: true,
    };

    setAttachments([newAttachment]);
    setErrorMessage(null);

    try {
      const resp = await ingestUrl(url, activeConversationId || undefined);
      setAttachments((prev) =>
        prev.map((att) =>
          att.id === attachmentId
            ? {
                ...att,
                uploadState: 'ready',
                docId: resp.doc_id,
                filename: resp.filename,
                sizeBytes: resp.size_bytes,
                pageCount: 1,
                fileType: 'url',
                isUrl: true,
              }
            : att
        )
      );
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Failed to ingest URL.';
      setAttachments((prev) =>
        prev.map((att) =>
          att.id === attachmentId
            ? {
                ...att,
                uploadState: 'error',
                errorMessage: errMsg,
              }
            : att
        )
      );
      setErrorMessage(errMsg);
    }
  };

  const handleRemoveAttachment = (id: string) => {
    setAttachments((prev) => prev.filter((att) => att.id !== id));
  };

  // Submit User Message
  const handleSendMessage = async (message: string, isRetry = false) => {
    if (!message.trim() || status === 'planning' || status === 'analyzing' || status === 'executing') {
      return;
    }

    setErrorMessage(null);
    setPlanData(null);
    setApprovalResult(null);
    setRejectionResult(null);
    setPdfAnalysisResult(null);
    setLastPrompt(message.trim());

    // 1. Ensure conversation exists upfront so the sidebar updates immediately
    let currentConvId = activeConversationId;
    if (!currentConvId) {
      try {
        const newConv = await createConversation('New Chat');
        currentConvId = newConv.id;
        setActiveConversationId(newConv.id);
        await loadConversations(newConv.id);
      } catch (err) {
        console.warn('Could not create conversation upfront, backend will assign one:', err);
      }
    }

    // 2. Only add optimistic user bubble if not a retry
    if (!isRetry) {
      const tempUserMsg: MessageItem = {
        id: `temp-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        role: 'user',
        content: message.trim(),
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, tempUserMsg]);
    }

    const activeAttachment = attachments.find((att) => att.uploadState === 'ready' && att.docId);

    // CASE A: Document Q&A
    if (attachments.length > 0) {
      if (!activeAttachment || !activeAttachment.docId) {
        setErrorMessage('Please wait for the document to finish uploading before sending.');
        return;
      }

      setAttachments((prev) =>
        prev.map((att) =>
          att.id === activeAttachment.id ? { ...att, uploadState: 'analyzing' } : att
        )
      );
      setStatus('analyzing');
      try {
        const analysis = await analyzePdf(activeAttachment.docId, message.trim(), currentConvId || undefined);
        setPdfAnalysisResult(analysis);
        setStatus('analysis_complete');
        setAttachments((prev) =>
          prev.map((att) =>
            att.id === activeAttachment.id ? { ...att, uploadState: 'ready' } : att
          )
        );

        const targetCid = analysis.conversation_id || currentConvId;
        if (targetCid) {
          if (targetCid !== activeConversationId) {
            setActiveConversationId(targetCid);
          }
          const detail = await fetchConversation(targetCid);
          setMessages(detail.messages || []);
          await loadConversations(targetCid);
        }
      } catch (err: any) {
        setStatus('error');
        setErrorMessage(err instanceof Error ? err.message : 'Document analysis failed.');
        setAttachments((prev) =>
          prev.map((att) =>
            att.id === activeAttachment.id ? { ...att, uploadState: 'ready' } : att
          )
        );
        const failedCid = err?.conversationId || currentConvId;
        if (failedCid) {
          await loadConversations(failedCid);
        }
      }
      return;
    }

    // CASE B: Standard / Planning Chat
    setStatus('planning');
    try {
      const resp = await sendChatMessage(message.trim(), currentConvId || undefined);
      setPlanData(resp);

      const targetCid = resp.conversation_id || currentConvId;
      if (targetCid && targetCid !== activeConversationId) {
        setActiveConversationId(targetCid);
      }

      if (resp.supported && resp.status === 'pending_approval') {
        setStatus('pending_approval');
      } else {
        // Supported general question answered or unsupported request
        setStatus('idle');
      }

      // Sync official SQLite messages
      if (targetCid) {
        try {
          const detail = await fetchConversation(targetCid);
          setMessages(detail.messages || []);
        } catch {
          // If fetch fails, append assistant message locally
          setMessages((prev) => [
            ...prev,
            {
              id: `asst-${Date.now()}`,
              role: 'assistant',
              content: resp.message,
              created_at: new Date().toISOString(),
            },
          ]);
        }
        await loadConversations(targetCid);
      }
    } catch (err: any) {
      setStatus('error');
      const msg = err instanceof Error ? err.message : 'Failed to generate plan.';
      setErrorMessage(msg);

      const failedCid = err?.conversationId || currentConvId;
      if (failedCid) {
        if (failedCid !== activeConversationId) {
          setActiveConversationId(failedCid);
        }
        await loadConversations(failedCid);
      }
    }
  };

  // Retry last failed prompt
  const handleRetryLastMessage = () => {
    if (!lastPrompt || status === 'planning' || status === 'analyzing') return;
    handleSendMessage(lastPrompt, true);
  };

  // Human Approval (Approve -> Act -> Verify)
  const handleApprove = async (actionId: string) => {
    if (!actionId) return;
    setActiveButton('approve');
    setStatus('executing');
    setErrorMessage(null);

    try {
      const resp = await approveAction(actionId);
      setApprovalResult(resp);
      setStatus('completed');
      if (activeConversationId) {
        const detail = await fetchConversation(activeConversationId);
        setMessages(detail.messages || []);
      }
    } catch (err) {
      setStatus('error');
      setErrorMessage(err instanceof Error ? err.message : 'Execution or verification failed.');
    } finally {
      setActiveButton(null);
    }
  };

  // Human Rejection (Gate stops execution)
  const handleReject = async (actionId: string) => {
    if (!actionId) return;
    setActiveButton('reject');
    setErrorMessage(null);

    try {
      const resp = await rejectAction(actionId);
      setRejectionResult(resp);
      setStatus('rejected');
      if (activeConversationId) {
        const detail = await fetchConversation(activeConversationId);
        setMessages(detail.messages || []);
      }
    } catch (err) {
      setStatus('error');
      setErrorMessage(err instanceof Error ? err.message : 'Failed to reject action.');
    } finally {
      setActiveButton(null);
    }
  };

  // Overwrite Confirmation
  const handleConfirmOverwrite = async (token: string, targetFile?: string) => {
    if (!token) return;
    setActiveButton('confirm_overwrite');
    setErrorMessage(null);

    try {
      const resp = await confirmOverwrite(token, targetFile);
      setPlanData(resp);
      setStatus('pending_approval');
      if (activeConversationId) {
        const detail = await fetchConversation(activeConversationId);
        setMessages(detail.messages || []);
      }
    } catch (err) {
      setStatus('error');
      setErrorMessage(err instanceof Error ? err.message : 'Failed to confirm overwrite.');
    } finally {
      setActiveButton(null);
    }
  };

  const handleReset = () => {
    setStatus('idle');
    setPlanData(null);
    setApprovalResult(null);
    setRejectionResult(null);
    setPdfAnalysisResult(null);
    setErrorMessage(null);
  };

  const isInteractionPending =
    status === 'pending_approval' ||
    status === 'executing' ||
    status === 'analyzing' ||
    Boolean(planData?.status === 'confirmation_required' && planData?.confirmation_token);

  // Has messages in active conversation
  const hasMessages = messages.length > 0;
  // Show welcome card ONLY when active conversation is completely empty and idle
  const showWelcomeCard = !hasMessages && status === 'idle' && !planData && !pdfAnalysisResult;

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <Sidebar
        health={health}
        healthLoading={healthLoading}
        onRefreshHealth={checkBackendHealth}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        onRenameConversation={handleRenameConversation}
        onDeleteConversation={handleDeleteConversation}
        disabled={isInteractionPending}
      />

      {/* Main Execution Canvas */}
      <main className="main-content">
        <div className="chat-viewport" ref={chatScrollRef}>
          {/* Slim Compact Timeline Visualizer */}
          <Timeline status={status} />

          {/* Welcome Card if brand new empty chat */}
          {showWelcomeCard && (
            <div className="welcome-hero">
              <div className="hero-icon">
                <ShieldCheck size={26} />
              </div>
              <div>
                <h2>KAIROS Permission Gate Active</h2>
                <p>
                  KAIROS uses <strong>hosted Gemma 4 AI</strong> to formulate structured plans and write safe text files.
                  Unlike standard bots, KAIROS strictly pauses at the <strong>Approve</strong> step to show you the exact action and file preview.
                  No file is ever written until you click Approve.
                </p>
                <p style={{ marginTop: '0.65rem', color: 'var(--text-muted)' }}>
                  💡 <em>Attach a syllabus or exam paper using <strong>+</strong>, click the <strong>Microphone</strong> to speak, or ask any question!</em>
                </p>
              </div>
            </div>
          )}

          {/* Loading conversation history indicator */}
          {historyLoading && (
            <div className="status-text text-muted" style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.8rem' }}>
              Loading conversation history...
            </div>
          )}

          {/* Render Persistent Saved Messages History */}
          {hasMessages && (
            <div className="chat-history-flow">
              {messages.map((msg) => {
                const isUser = msg.role === 'user';
                const msgId = msg.id;

                return (
                  <div key={msgId} className={`message-row ${isUser ? 'user-row' : 'assistant-row'}`}>
                    <div className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
                      <div className="bubble-sender">
                        <div className="bubble-sender-title">
                          {isUser ? (
                            <>
                              <User size={12} className="bubble-sender-icon" />
                              <span>You</span>
                            </>
                          ) : (
                            <>
                              <Sparkles size={12} className="bubble-sender-icon text-cyan" />
                              <span>KAIROS Assistant</span>
                            </>
                          )}
                        </div>
                        <span className="bubble-timestamp">
                          {new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      </div>

                      <div className="bubble-content">
                        {isUser ? (
                          <div className="bubble-text">{msg.content}</div>
                        ) : (
                          <div className="markdown-answer-body">
                            <MarkdownRenderer content={msg.content} />
                          </div>
                        )}
                      </div>

                      {!isUser && (
                        <div className="message-toolbar">
                          <button
                            type="button"
                            className="msg-tool-btn"
                            onClick={() => handleCopyText(msg.content, msgId)}
                            title="Copy message"
                          >
                            {copiedId === msgId ? (
                              <>
                                <Check size={12} className="text-emerald" />
                                <span>Copied</span>
                              </>
                            ) : (
                              <>
                                <Copy size={12} />
                                <span>Copy</span>
                              </>
                            )}
                          </button>

                          <button
                            type="button"
                            className={`msg-tool-btn ${speakingId === msgId ? 'speaking-active' : ''}`}
                            onClick={() => handleToggleSpeak(msg.content, msgId)}
                            title={speakingId === msgId ? 'Stop speaking' : 'Read aloud (en-IN)'}
                          >
                            {speakingId === msgId ? (
                              <>
                                <VolumeX size={12} className="text-cyan animate-pulse" />
                                <span>Stop</span>
                              </>
                            ) : (
                              <>
                                <Volume2 size={12} />
                                <span>Read Aloud</span>
                              </>
                            )}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}

              {/* Model Thinking Indicator while planning */}
              {status === 'planning' && (
                <div className="message-row assistant-row">
                  <div className="message-bubble assistant-bubble thinking-bubble">
                    <div className="bubble-sender">
                      <div className="bubble-sender-title">
                        <Sparkles size={12} className="bubble-sender-icon text-cyan animate-pulse" />
                        <span>KAIROS Assistant</span>
                      </div>
                    </div>
                    <div className="thinking-content">
                      <Loader2 size={15} className="animate-spin text-cyan" />
                      <span>Gemma is thinking and formulating a plan...</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Model Analyzing Indicator while analyzing document or image */}
              {status === 'analyzing' && (
                <div className="message-row assistant-row">
                  <div className="message-bubble assistant-bubble thinking-bubble">
                    <div className="bubble-sender">
                      <div className="bubble-sender-title">
                        <Sparkles size={12} className="bubble-sender-icon text-cyan animate-pulse" />
                        <span>KAIROS Assistant</span>
                      </div>
                    </div>
                    <div className="thinking-content">
                      <Loader2 size={15} className="animate-spin text-cyan" />
                      <span>
                        {attachments.some((a) => a.fileType === 'url' || a.isUrl)
                          ? 'Extracting webpage text and analyzing with Gemma...'
                          : attachments.some((a) => a.fileType === 'image' || a.isImage)
                          ? 'Extracting image text with OCR and analyzing with Gemma...'
                          : 'Extracting document text and analyzing with Gemma...'}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Active Turn Overwrite Confirmation Card */}
          {planData && planData.status === 'confirmation_required' && planData.confirmation_token && (
            <div className="plan-card confirmation-card">
              <div className="plan-card-header">
                <div className="header-left">
                  <AlertTriangle className="warning-icon text-amber" size={20} />
                  <div>
                    <div className="title-row">
                      <h3 className="plan-title">Overwrite Confirmation Required</h3>
                      <span className="engine-badge badge-demo">Existing File</span>
                    </div>
                    <p className="plan-subtitle">
                      KAIROS detected an existing file: <code>workspace/{planData.confirmation_target_file}</code>
                    </p>
                  </div>
                </div>
              </div>
              <div className="demo-disclaimer" style={{ background: 'rgba(245, 158, 11, 0.08)', borderColor: 'rgba(245, 158, 11, 0.25)', color: 'var(--amber, #f59e0b)' }}>
                <AlertTriangle size={16} className="disclaimer-icon" />
                <span>{planData.message}</span>
              </div>
              <div className="plan-actions-footer">
                <button className="btn-reject" onClick={handleReset} disabled={activeButton !== null}>
                  <X size={16} />
                  <span>Cancel</span>
                </button>
                <button
                  className="btn-approve"
                  onClick={() => handleConfirmOverwrite(planData.confirmation_token!, planData.confirmation_target_file || undefined)}
                  disabled={activeButton !== null}
                >
                  {activeButton === 'confirm_overwrite' ? (
                    <>
                      <Loader2 size={16} className="animate-spin" />
                      <span>Confirming...</span>
                    </>
                  ) : (
                    <>
                      <Check size={16} />
                      <span>Confirm Overwrite</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Active Turn Plan Card (Pending Approval) */}
          {status === 'pending_approval' && planData && (
            <PlanCard
              planData={planData}
              onApprove={handleApprove}
              onReject={handleReject}
              isLoading={activeButton !== null}
              activeAction={activeButton === 'confirm_overwrite' ? null : activeButton}
            />
          )}

          {/* Active Turn PDF Analysis Response Card */}
          {pdfAnalysisResult && (
            <div className="result-card result-success pdf-response-card">
              <div className="result-header">
                <div className="icon-badge badge-gemma">
                  <Sparkles size={22} />
                </div>
                <div>
                  <div className="title-row">
                    <h3 className="result-title">KAIROS Analysis</h3>
                    <span className="engine-badge badge-gemma">Gemma 4 AI</span>
                    <span className="pdf-doc-badge">
                      {pdfAnalysisResult.is_url ? (
                        <Globe size={12} />
                      ) : pdfAnalysisResult.is_image ? (
                        <ImageIcon size={12} />
                      ) : (
                        <FileText size={12} />
                      )}{' '}
                      {pdfAnalysisResult.filename}{' '}
                      {pdfAnalysisResult.is_url
                        ? '(Webpage)'
                        : pdfAnalysisResult.is_image
                        ? '(Image)'
                        : `(${pdfAnalysisResult.total_pages} pg)`}
                    </span>
                    {pdfAnalysisResult.is_url ? (
                      <span className="pdf-ocr-badge">Web Content</span>
                    ) : pdfAnalysisResult.is_image ? (
                      <span className="pdf-ocr-badge">Local OCR</span>
                    ) : (
                      <>
                        {pdfAnalysisResult.is_mixed && (
                          <span className="pdf-ocr-badge">Hybrid (Native + OCR)</span>
                        )}
                        {pdfAnalysisResult.is_scanned && pdfAnalysisResult.ocr_available && (
                          <span className="pdf-ocr-badge">Local OCR</span>
                        )}
                        {pdfAnalysisResult.is_scanned && !pdfAnalysisResult.ocr_available && (
                          <span className="pdf-ocr-badge badge-warning">OCR Missing</span>
                        )}
                      </>
                    )}
                  </div>
                  <p className="result-message">
                    {pdfAnalysisResult.is_url
                      ? 'Read-only analysis grounded in securely fetched webpage text.'
                      : pdfAnalysisResult.is_image
                      ? 'Read-only analysis grounded in OCR-extracted image text.'
                      : 'Read-only analysis grounded in extracted document text.'}
                  </p>
                </div>
              </div>

              <div className="result-body">
                <div className="pdf-answer-text">
                  <MarkdownRenderer content={pdfAnalysisResult.answer} />
                </div>
              </div>

              <div className="result-footer pdf-card-footer">
                <button
                  type="button"
                  className="btn-copy-answer"
                  onClick={() => handleCopyText(pdfAnalysisResult.answer, 'active-pdf')}
                  title="Copy answer"
                >
                  {copiedId === 'active-pdf' ? (
                    <>
                      <Check size={14} className="text-emerald" />
                      <span>Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy size={14} />
                      <span>Copy Answer</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  className={`btn-speak-answer ${speakingId === 'active-pdf' ? 'speaking-active' : ''}`}
                  onClick={() => handleToggleSpeak(pdfAnalysisResult.answer, 'active-pdf')}
                  title="Read answer aloud"
                >
                  {speakingId === 'active-pdf' ? (
                    <>
                      <VolumeX size={14} className="text-cyan animate-pulse" />
                      <span>Stop Speaking</span>
                    </>
                  ) : (
                    <>
                      <Volume2 size={14} />
                      <span>Read Aloud</span>
                    </>
                  )}
                </button>

                <button className="btn-reset" onClick={handleReset}>
                  Ask another question
                </button>
              </div>
            </div>
          )}

          {/* Execution or Rejection Result Card */}
          {!pdfAnalysisResult &&
            (status === 'completed' || status === 'rejected') && (
              <ExecutionResult
                status={status}
                approvalResult={approvalResult}
                rejectionResult={rejectionResult}
                unsupportedMessage={planData?.message}
                errorMessage={errorMessage}
                onReset={handleReset}
              />
            )}

          {/* Inline Actionable Error Card with Retry Button */}
          {status === 'error' && errorMessage && (
            <div className="chat-error-card" role="alert">
              <div className="error-card-header">
                <AlertCircle size={20} className="error-card-icon" />
                <div className="error-card-text">
                  <h4 className="error-card-title">Request Could Not Be Completed</h4>
                  <p className="error-card-desc">{errorMessage}</p>
                </div>
              </div>
              <div className="error-card-actions">
                {lastPrompt && (
                  <button
                    type="button"
                    className="btn-retry-action"
                    onClick={handleRetryLastMessage}
                    title="Retry sending the prompt"
                  >
                    <RotateCcw size={13} />
                    <span>Retry Request</span>
                  </button>
                )}
                <button
                  type="button"
                  className="btn-dismiss-error"
                  onClick={() => {
                    setStatus('idle');
                    setErrorMessage(null);
                  }}
                >
                  Dismiss
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Input Bar with Attachment and Voice support */}
        <footer className="chat-input-container">
          <ChatInput
            onSendMessage={handleSendMessage}
            isLoading={status === 'planning' || status === 'analyzing'}
            disabled={isInteractionPending}
            attachments={attachments}
            onAddAttachment={handleAddAttachment}
            onAddUrl={handleAddUrl}
            onRemoveAttachment={handleRemoveAttachment}
          />
        </footer>
      </main>
    </div>
  );
};
export default App;
