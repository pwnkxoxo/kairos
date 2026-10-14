import React, { useState, useEffect, useRef } from 'react';
import {
  Send,
  Loader2,
  Sparkles,
  Plus,
  FileText,
  Image as ImageIcon,
  X,
  AlertCircle,
  CheckCircle2,
  BookOpen,
  CheckSquare,
  HelpCircle,
  LucideIcon,
  Mic,
  MicOff,
  MapPin,
  Globe,
} from 'lucide-react';
import { AttachmentItem } from '../types/api';

interface QuickAction {
  id: string;
  label: string;
  prompt: string;
  icon: LucideIcon;
}

const PDF_QUICK_ACTIONS: QuickAction[] = [
  {
    id: 'revision_map',
    label: 'Revision Map',
    prompt: 'Generate an Exam Revision Map from the attached syllabus and exam papers: identify which syllabus topics recur in exam questions with [Page X] citations, which topics have no exam evidence, and which questions are unverified.',
    icon: MapPin,
  },
  {
    id: 'summarize_pdf',
    label: 'Summarize',
    prompt: 'Summarise this document comprehensively, highlighting the core objectives, findings, and key takeaways with source page citations.',
    icon: FileText,
  },
  {
    id: 'explain_pdf',
    label: 'Explain simply',
    prompt: 'Explain the concepts in this document in simple, beginner-friendly language with clear analogies and step-by-step explanations.',
    icon: Sparkles,
  },
  {
    id: 'notes_pdf',
    label: 'Revision notes',
    prompt: 'Create concise, high-yield revision notes and bullet-point summaries from this document, citing relevant pages.',
    icon: BookOpen,
  },
  {
    id: 'quiz_pdf',
    label: 'Practice quiz',
    prompt: 'Generate a practice quiz with 5 multiple-choice questions based on this document, with correct answers and explanations citing specific pages.',
    icon: HelpCircle,
  },
  {
    id: 'checklist_pdf',
    label: 'Action checklist',
    prompt: 'Extract an actionable checklist of key steps, recommendations, and important action items from this document.',
    icon: CheckSquare,
  },
];

const URL_QUICK_ACTIONS: QuickAction[] = [
  {
    id: 'summarize_url',
    label: 'Summarize webpage',
    prompt: 'Summarise this webpage comprehensively, highlighting the core message, key arguments, and takeaways.',
    icon: FileText,
  },
  {
    id: 'takeaways_url',
    label: 'Key takeaways',
    prompt: 'Extract the most important key points and takeaways from this webpage in concise bullet points.',
    icon: BookOpen,
  },
  {
    id: 'explain_url',
    label: 'Explain simply',
    prompt: 'Explain the concepts on this webpage in simple, beginner-friendly language with clear analogies and step-by-step explanations.',
    icon: Sparkles,
  },
  {
    id: 'checklist_url',
    label: 'Action checklist',
    prompt: 'Extract an actionable checklist of key steps, recommendations, and important action items from this webpage.',
    icon: CheckSquare,
  },
];

const IMAGE_QUICK_ACTIONS: QuickAction[] = [
  {
    id: 'extract_image',
    label: 'Extract text',
    prompt: 'Extract and transcribe all readable text from this image.',
    icon: FileText,
  },
  {
    id: 'summarize_image',
    label: 'Summarize',
    prompt: 'Summarise the text extracted from this image, highlighting the main takeaways.',
    icon: Sparkles,
  },
  {
    id: 'explain_image',
    label: 'Explain simply',
    prompt: 'Explain the concepts and details in the text extracted from this image in simple, beginner-friendly language.',
    icon: BookOpen,
  },
  {
    id: 'checklist_image',
    label: 'Action checklist',
    prompt: 'Extract an actionable checklist of key steps and items from the text in this image.',
    icon: CheckSquare,
  },
];

const MIXED_QUICK_ACTIONS: QuickAction[] = [
  {
    id: 'summarize_mixed',
    label: 'Summarize all',
    prompt: 'Summarise the core content, findings, and key takeaways across the attached sources.',
    icon: FileText,
  },
  {
    id: 'takeaways_mixed',
    label: 'Key takeaways',
    prompt: 'Extract the most important key points and takeaways across the attached sources in concise bullet points.',
    icon: BookOpen,
  },
  {
    id: 'explain_mixed',
    label: 'Explain simply',
    prompt: 'Explain the main concepts from the attached sources in simple, beginner-friendly language with clear analogies.',
    icon: Sparkles,
  },
  {
    id: 'checklist_mixed',
    label: 'Action checklist',
    prompt: 'Extract an actionable checklist of key steps, recommendations, and important action items from the attached sources.',
    icon: CheckSquare,
  },
];

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  disabled: boolean;
  presetMessage?: string;
  attachments?: AttachmentItem[];
  onAddAttachment?: (file: File) => void;
  onAddUrl?: (url: string) => void;
  onRemoveAttachment?: (id: string) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading,
  disabled,
  presetMessage,
  attachments = [],
  onAddAttachment,
  onAddUrl,
  onRemoveAttachment,
}) => {
  const [input, setInput] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // URL modal state
  const [showUrlModal, setShowUrlModal] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [urlModalError, setUrlModalError] = useState<string | null>(null);

  // Speech Recognition state (Voice interface)
  const [isRecording, setIsRecording] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (presetMessage) {
      setInput(presetMessage);
    }
  }, [presetMessage]);

  // Clean up recognition on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch {
          // ignore
        }
      }
    };
  }, []);

  const toggleRecording = () => {
    setVoiceError(null);
    if (isRecording) {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch {
          // ignore
        }
      }
      setIsRecording(false);
      return;
    }

    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setVoiceError(
        'Speech recognition is not supported in this browser. Please use Chrome, Edge, or enter text manually.'
      );
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = 'en-IN';
      recognition.continuous = false;
      recognition.interimResults = true;
      recognitionRef.current = recognition;

      recognition.onstart = () => {
        setIsRecording(true);
        setVoiceError(null);
      };

      recognition.onresult = (event: any) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          transcript += event.results[i][0].transcript;
        }
        if (transcript.trim()) {
          setInput((prev) => (prev ? `${prev} ${transcript.trim()}` : transcript.trim()));
        }
      };

      recognition.onerror = (event: any) => {
        if (recognitionRef.current) {
          try {
            recognitionRef.current.abort();
          } catch {
            // ignore
          }
        }
        setIsRecording(false);

        const errorCode = event.error;
        if (errorCode === 'network') {
          setVoiceError(
            'Speech recognition service could not connect. Check internet connectivity, or note that privacy-focused browsers (like Brave) or offline environments disable remote speech services. Typed input remains fully functional.'
          );
        } else if (errorCode === 'not-allowed') {
          setVoiceError('Microphone permission denied. Please allow microphone access in your browser settings.');
        } else if (errorCode === 'service-not-allowed') {
          setVoiceError('Speech recognition service is blocked by browser policy or platform restrictions. Typed input remains fully functional.');
        } else if (errorCode === 'audio-capture') {
          setVoiceError('No usable microphone was detected. Please check your system audio input device.');
        } else if (errorCode === 'no-speech') {
          setVoiceError('No speech was detected. You can click the microphone to try speaking again, or type your request.');
        } else {
          setVoiceError(`Voice input notice (${errorCode}). Typed input remains fully functional.`);
        }
      };

      recognition.onend = () => {
        setIsRecording(false);
      };

      recognition.start();
    } catch (err: any) {
      setIsRecording(false);
      setVoiceError(`Failed to start recording: ${err.message || 'Unknown error'}`);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isRecording && recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        // ignore
      }
      setIsRecording(false);
    }
    if (!input.trim() || isLoading || disabled) return;
    onSendMessage(input.trim());
    setInput('');
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && onAddAttachment) {
      onAddAttachment(file);
    }
    // Reset file input so user can select the same file again if desired
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleTriggerQuickAction = (prompt: string) => {
    if (isLoading || disabled) return;
    onSendMessage(prompt);
    setInput('');
  };

  const hasAttachments = attachments.length > 0;
  const isBusy = attachments.some(
    (att) => att.uploadState !== 'ready' && att.uploadState !== 'error'
  );
  const activeAttachment = attachments.find((att) => att.uploadState === 'ready');

  // Multi-attachment and source-type classification
  const readyAttachments = attachments.filter((att) => att.uploadState === 'ready');
  const targetAttachments = readyAttachments.length > 0 ? readyAttachments : attachments;

  const hasUrl = targetAttachments.some((att) => att.fileType === 'url' || att.isUrl);
  const hasImage = targetAttachments.some((att) => att.fileType === 'image' || att.isImage);
  const hasPdf = targetAttachments.some(
    (att) => att.fileType === 'pdf' || (!att.isUrl && !att.isImage)
  );

  const distinctTypesCount = [hasUrl, hasImage, hasPdf].filter(Boolean).length;
  const isMixed = distinctTypesCount > 1;

  type SourceType = 'url' | 'image' | 'pdf' | 'mixed';
  let activeSourceType: SourceType = 'pdf';
  if (isMixed) {
    activeSourceType = 'mixed';
  } else if (hasUrl) {
    activeSourceType = 'url';
  } else if (hasImage) {
    activeSourceType = 'image';
  } else {
    activeSourceType = 'pdf';
  }

  const getSourceLabel = (): string => {
    switch (activeSourceType) {
      case 'url':
        return 'Webpage';
      case 'image':
        return 'Image';
      case 'mixed':
        return 'Sources';
      case 'pdf':
      default:
        return 'Document';
    }
  };

  const sourceLabel = getSourceLabel();

  const getQuickActions = (): QuickAction[] => {
    switch (activeSourceType) {
      case 'url':
        return URL_QUICK_ACTIONS;
      case 'image':
        return IMAGE_QUICK_ACTIONS;
      case 'mixed':
        return MIXED_QUICK_ACTIONS;
      case 'pdf':
      default:
        return PDF_QUICK_ACTIONS;
    }
  };

  const quickActions = getQuickActions();

  const getStatusText = (att: AttachmentItem) => {
    const isImg = att.fileType === 'image' || att.isImage;
    const isUrl = att.fileType === 'url' || att.isUrl;
    switch (att.uploadState) {
      case 'uploading':
        if (isUrl) return 'Fetching & reading webpage...';
        return isImg ? 'Uploading image...' : 'Uploading PDF...';
      case 'extracting_text':
        return 'Extracting text...';
      case 'reading_ocr':
        return isImg ? 'Reading image with OCR...' : 'Reading scanned pages with OCR...';
      case 'analyzing':
        if (isUrl) return 'Analysing webpage text...';
        return isImg ? 'Analysing image text...' : 'Analysing document...';
      case 'ready':
        if (isUrl) {
          return `${(att.sizeBytes / 1024).toFixed(0)} KB • Web Page`;
        }
        if (isImg) {
          return `${(att.sizeBytes / 1024).toFixed(0)} KB • Image (OCR)`;
        }
        return `${(att.sizeBytes / 1024).toFixed(0)} KB${
          att.pageCount ? ` • ${att.pageCount} pg` : ''
        }${att.isScanned ? ' (Scanned/OCR)' : ''}`;
      case 'error':
        return 'Upload error';
      default:
        return 'Processing...';
    }
  };

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = urlInput.trim();
    if (!trimmed) {
      setUrlModalError('Please enter a URL.');
      return;
    }
    if (!trimmed.startsWith('http://') && !trimmed.startsWith('https://')) {
      setUrlModalError('URL must begin with http:// or https://');
      return;
    }
    setUrlModalError(null);
    setShowUrlModal(false);
    setUrlInput('');
    if (onAddUrl) {
      onAddUrl(trimmed);
    }
  };

  const placeholderText = () => {
    if (isRecording) {
      return 'Listening... speak clearly into your microphone...';
    }
    if (disabled) {
      return 'Resolve the pending plan above first...';
    }
    if (hasAttachments) {
      if (activeSourceType === 'mixed') {
        return 'Ask a question about the attached sources (or select a quick action above)...';
      }
      const targetName = activeAttachment?.filename || attachments[0]?.filename;
      if (targetName) {
        return `Ask a question about ${targetName} (or select a quick action above)...`;
      }
      return `Ask a question about the attached ${sourceLabel.toLowerCase()} (or select a quick action above)...`;
    }
    return 'Instruct KAIROS or click microphone to speak...';
  };

  return (
    <form className="chat-input-form" onSubmit={handleSubmit}>
      {/* Hidden file input for PDF or image attachments */}
      <input
        type="file"
        ref={fileInputRef}
        accept=".pdf,application/pdf,image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {/* URL Ingestion Modal */}
      {showUrlModal && (
        <div className="url-modal-backdrop" onClick={() => setShowUrlModal(false)}>
          <div className="url-modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="url-modal-header">
              <div className="url-modal-title">
                <Globe size={18} className="text-cyan" />
                <span>Read Public Webpage</span>
              </div>
              <button
                type="button"
                className="url-modal-close"
                onClick={() => {
                  setShowUrlModal(false);
                  setUrlModalError(null);
                }}
                aria-label="Close dialog"
              >
                <X size={16} />
              </button>
            </div>
            <p className="url-modal-desc">
              Enter a public webpage URL. KAIROS will securely fetch and parse its text content for read-only Q&amp;A.
            </p>
            <div className="url-modal-input-row">
              <input
                type="url"
                className="url-modal-input"
                placeholder="https://example.com/article"
                value={urlInput}
                onChange={(e) => {
                  setUrlInput(e.target.value);
                  if (urlModalError) setUrlModalError(null);
                }}
                autoFocus
              />
              <button
                type="button"
                className="url-modal-submit-btn"
                onClick={handleUrlSubmit}
                disabled={!urlInput.trim()}
              >
                Attach URL
              </button>
            </div>
            {urlModalError && (
              <p className="url-modal-error">
                <AlertCircle size={13} />
                <span>{urlModalError}</span>
              </p>
            )}
          </div>
        </div>
      )}

      {/* Voice Status Alert if error or recording */}
      {voiceError && (
        <div className="voice-alert-banner error" role="alert">
          <div className="voice-alert-content">
            <AlertCircle size={15} className="voice-icon" />
            <span>{voiceError}</span>
          </div>
          <button
            type="button"
            onClick={() => setVoiceError(null)}
            className="voice-alert-close"
            aria-label="Dismiss voice message"
            title="Dismiss voice message"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {isRecording && (
        <div className="voice-alert-banner recording" role="status">
          <span className="voice-pulse-dot" />
          <span>Recording speech (en-IN). Words will appear below to edit. (Note: browser may process audio via remote speech service).</span>
          <button type="button" className="voice-stop-btn" onClick={toggleRecording}>
            Done
          </button>
        </div>
      )}

      {/* Attachment Tray (renders compact chip above input) */}
      {hasAttachments && (
        <div className="attachments-tray" role="region" aria-label="Attached files">
          {attachments.map((att) => (
            <div
              key={att.id}
              className={`attachment-chip chip-${att.uploadState}`}
              title={att.errorMessage || att.filename}
            >
              <div className="attachment-chip-icon">
                {att.fileType === 'url' || att.isUrl ? (
                  <Globe size={15} />
                ) : att.fileType === 'image' || att.isImage ? (
                  <ImageIcon size={15} />
                ) : (
                  <FileText size={15} />
                )}
              </div>
              <div className="attachment-chip-info">
                <span className="attachment-chip-name">{att.filename}</span>
                <span className="attachment-chip-meta">{getStatusText(att)}</span>
              </div>

              {att.uploadState !== 'ready' && att.uploadState !== 'error' && (
                <Loader2 size={14} className="animate-spin chip-spinner" />
              )}
              {att.uploadState === 'ready' && (
                <CheckCircle2 size={14} className="chip-ready-icon" />
              )}
              {att.uploadState === 'error' && (
                <AlertCircle size={14} className="chip-error-icon" />
              )}

              <button
                type="button"
                className="attachment-chip-remove"
                onClick={() => onRemoveAttachment?.(att.id)}
                disabled={isLoading}
                title="Remove attachment"
                aria-label={`Remove ${att.filename}`}
              >
                <X size={13} />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Quick Actions Tray (visible when attachment is ready) */}
      {hasAttachments && activeAttachment && (
        <div
          className="quick-actions-bar"
          role="toolbar"
          aria-label={`Quick ${activeSourceType === 'mixed' ? 'source' : sourceLabel.toLowerCase()} actions`}
        >
          <span className="quick-actions-label">Quick Actions:</span>
          <div className="quick-actions-chips">
            {quickActions.map((action) => {
              const ActionIcon = action.icon;
              return (
                <button
                  key={action.id}
                  type="button"
                  className="quick-action-btn"
                  onClick={() => handleTriggerQuickAction(action.prompt)}
                  disabled={isLoading || disabled || isBusy}
                  title={`Run: ${action.label}`}
                >
                  <ActionIcon size={12} className="quick-action-icon" />
                  <span>{action.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className={`input-wrapper ${hasAttachments ? 'has-attachments' : ''} ${isRecording ? 'is-recording' : ''}`}>
        {/* Plus (+) Button to attach PDF or Image */}
        <button
          type="button"
          className="attach-button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading || disabled || isBusy}
          title="Attach PDF (Max 10 MB) or image (PNG/JPEG/WebP, Max 5 MB)"
          aria-label="Attach PDF or image"
        >
          <Plus size={18} />
        </button>

        {/* Globe Button to read webpage URL */}
        <button
          type="button"
          className="attach-button attach-url-button"
          onClick={() => setShowUrlModal(true)}
          disabled={isLoading || disabled || isBusy}
          title="Read public webpage URL"
          aria-label="Read webpage URL"
        >
          <Globe size={17} />
        </button>

        {/* Microphone Button for Voice Interface */}
        <button
          type="button"
          className={`mic-button ${isRecording ? 'recording-active' : ''}`}
          onClick={toggleRecording}
          disabled={isLoading || disabled}
          title={isRecording ? 'Stop recording' : 'Speak instruction (en-IN)'}
          aria-label={isRecording ? 'Stop recording' : 'Start voice input'}
        >
          {isRecording ? <MicOff size={18} className="mic-active-icon animate-pulse" /> : <Mic size={18} />}
        </button>

        <Sparkles size={17} className="input-icon" />

        <input
          type="text"
          className="chat-input"
          placeholder={placeholderText()}
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
            if (voiceError) setVoiceError(null);
          }}
          disabled={isLoading || disabled || isBusy}
        />

        <button
          type="submit"
          className="send-button"
          disabled={!input.trim() || isLoading || disabled || isBusy}
        >
          {isLoading ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <>
              <span>Send</span>
              <Send size={14} />
            </>
          )}
        </button>
      </div>

      <p className="input-hint">
        {hasAttachments ? (
          <>
            {sourceLabel} attached. Ask custom questions or select a <strong>Quick Action</strong> above.
          </>
        ) : (
          <>
            Press <strong>Enter</strong> to submit. KAIROS will formulate a plan and request your explicit approval.
          </>
        )}
      </p>
    </form>
  );
};
