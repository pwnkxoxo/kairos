import React, { useState } from 'react';
import { ChatResponse } from '../types/api';
import { ShieldAlert, FileCode2, Check, X, Loader2, Sparkles, ChevronDown, ChevronUp, AlertTriangle } from 'lucide-react';

interface PlanCardProps {
  planData: ChatResponse;
  onApprove: (actionId: string) => void;
  onReject: (actionId: string) => void;
  isLoading: boolean;
  activeAction: 'approve' | 'reject' | null;
}

export const PlanCard: React.FC<PlanCardProps> = ({
  planData,
  onApprove,
  onReject,
  isLoading,
  activeAction,
}) => {
  const [showPreview, setShowPreview] = useState(true);
  const isGemma = planData.mode === 'gemma';
  const actionId = planData.action_id || '';

  return (
    <div className="plan-card">
      <div className="plan-card-header">
        <div className="header-left">
          <ShieldAlert className="warning-icon" size={20} />
          <div>
            <div className="title-row">
              <h3 className="plan-title">Action Approval Required</h3>
              <span className={`engine-badge ${isGemma ? 'badge-gemma' : 'badge-demo'}`}>
                <Sparkles size={12} />
                {isGemma ? 'Gemma 4 AI' : 'Demo Template'}
              </span>
            </div>
            <p className="plan-subtitle">
              KAIROS is requesting explicit human permission before writing to disk.
            </p>
          </div>
        </div>
        {actionId && (
          <span className="action-id-tag">ID: {actionId.slice(0, 8)}...</span>
        )}
      </div>

      {/* Model Message / Disclaimer Banner */}
      <div className={`demo-disclaimer ${isGemma ? 'disclaimer-gemma' : ''}`}>
        <Sparkles size={16} className="disclaimer-icon" />
        <span>{planData.ai_disclaimer}</span>
      </div>

      {/* Assistant Summary if available */}
      {planData.message && (
        <div className="assistant-summary-box">
          <p className="summary-label">KAIROS's Response:</p>
          <p className="summary-text">{planData.message}</p>
        </div>
      )}

      {/* Steps List */}
      <div className="plan-steps-list">
        {planData.plan.map((step) => (
          <div key={step.step_number} className="plan-step-item">
            <div className="step-badge">Step {step.step_number}</div>
            <div className="step-content">
              <div className="step-main-row">
                <span className="step-action-tag">{step.action}</span>
                <span className="step-target">
                  <FileCode2 size={14} /> workspace/{step.target_file}
                </span>
                {step.allow_overwrite && (
                  <span className="step-overwrite-tag">
                    <AlertTriangle size={13} />
                    Overwrites existing file
                  </span>
                )}
              </div>
              <p className="step-description">{step.description}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Validated File Content Preview */}
      {planData.file_preview && (
        <div className="content-preview-container">
          <button
            type="button"
            className="preview-toggle-btn"
            onClick={() => setShowPreview(!showPreview)}
          >
            <span>
              Preview intended content for{' '}
              <code>workspace/{planData.plan[0]?.target_file || 'demo.txt'}</code> ({planData.file_preview.length} chars)
            </span>
            {showPreview ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          {showPreview && (
            <pre className="content-preview-box">
              <code>{planData.file_preview}</code>
            </pre>
          )}
        </div>
      )}

      {/* Action Decision Buttons */}
      <div className="plan-actions-footer">
        <button
          className="btn-reject"
          onClick={() => onReject(actionId)}
          disabled={isLoading || !actionId}
        >
          {isLoading && activeAction === 'reject' ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              <span>Rejecting...</span>
            </>
          ) : (
            <>
              <X size={16} />
              <span>Reject Action</span>
            </>
          )}
        </button>

        <button
          className="btn-approve"
          onClick={() => onApprove(actionId)}
          disabled={isLoading || !actionId}
        >
          {isLoading && activeAction === 'approve' ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              <span>Executing safely...</span>
            </>
          ) : (
            <>
              <Check size={16} />
              <span>Approve & Execute</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
