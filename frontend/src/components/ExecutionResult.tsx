import React from 'react';
import { ApprovalResponse, RejectionResponse } from '../types/api';
import { CheckCircle2, RotateCcw, FileCheck, ShieldX, AlertCircle, ShieldAlert } from 'lucide-react';

interface ExecutionResultProps {
  status: 'completed' | 'rejected' | 'error' | 'unsupported';
  approvalResult: ApprovalResponse | null;
  rejectionResult: RejectionResponse | null;
  unsupportedMessage?: string | null;
  errorMessage: string | null;
  onReset: () => void;
}

export const ExecutionResult: React.FC<ExecutionResultProps> = ({
  status,
  approvalResult,
  rejectionResult,
  unsupportedMessage,
  errorMessage,
  onReset,
}) => {
  if (status === 'completed' && approvalResult) {
    return (
      <div className="result-card result-success">
        <div className="result-header">
          <div className="icon-badge badge-success">
            <CheckCircle2 size={24} />
          </div>
          <div>
            <h3 className="result-title">Action Approved & Verified</h3>
            <p className="result-message">{approvalResult.message}</p>
          </div>
        </div>

        <div className="result-body">
          <h4 className="details-heading">Verification Summary:</h4>
          {approvalResult.results.map((res, i) => (
            <div key={i} className="verification-item">
              <FileCheck size={16} className="text-emerald" />
              <div className="verification-info">
                <span className="file-path">{res.target_file}</span>
                <span className="file-meta">
                  Status: <strong>{res.status}</strong> • Size: {res.bytes_written} bytes
                </span>
              </div>
              <span className="verified-badge">Disk Verified</span>
            </div>
          ))}
        </div>

        <div className="result-footer">
          <button className="btn-reset" onClick={onReset}>
            <RotateCcw size={16} />
            <span>Start Another Request</span>
          </button>
        </div>
      </div>
    );
  }

  if (status === 'rejected' && rejectionResult) {
    return (
      <div className="result-card result-rejected">
        <div className="result-header">
          <div className="icon-badge badge-rejected">
            <ShieldX size={24} />
          </div>
          <div>
            <h3 className="result-title">Action Rejected by Human Gate</h3>
            <p className="result-message">{rejectionResult.message}</p>
          </div>
        </div>

        <div className="result-body">
          <div className="safety-notice">
            <p>
              🛡️ <strong>Zero Side-Effects:</strong> No files were created, modified, or executed
              inside the workspace or filesystem.
            </p>
          </div>
        </div>

        <div className="result-footer">
          <button className="btn-reset" onClick={onReset}>
            <RotateCcw size={16} />
            <span>Start Another Request</span>
          </button>
        </div>
      </div>
    );
  }

  if (status === 'unsupported') {
    return (
      <div className="result-card result-unsupported">
        <div className="result-header">
          <div className="icon-badge badge-unsupported">
            <ShieldAlert size={24} />
          </div>
          <div>
            <h3 className="result-title">Request Not Supported for Automated Action</h3>
            <p className="result-message">
              {unsupportedMessage ||
                'For safety, KAIROS only executes explicit file creation inside the workspace.'}
            </p>
          </div>
        </div>

        <div className="result-body">
          <div className="safety-notice">
            <p>
              🛡️ <strong>Safety Constraint:</strong> KAIROS never runs arbitrary code, terminal commands,
              or unrestricted filesystem modifications. Please ask KAIROS to create or write a text file.
            </p>
          </div>
        </div>

        <div className="result-footer">
          <button className="btn-reset" onClick={onReset}>
            <RotateCcw size={16} />
            <span>Try Another Request</span>
          </button>
        </div>
      </div>
    );
  }

  if (status === 'error' && errorMessage) {
    return (
      <div className="result-card result-error">
        <div className="result-header">
          <div className="icon-badge badge-error">
            <AlertCircle size={24} />
          </div>
          <div>
            <h3 className="result-title">Operation Failed</h3>
            <p className="result-message">{errorMessage}</p>
          </div>
        </div>

        <div className="result-footer">
          <button className="btn-reset" onClick={onReset}>
            <RotateCcw size={16} />
            <span>Try Again</span>
          </button>
        </div>
      </div>
    );
  }

  return null;
};
