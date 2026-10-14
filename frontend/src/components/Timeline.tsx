import React from 'react';
import { ActionStatus } from '../types/api';
import { Brain, FileText, Lock, Play, CheckCircle, XCircle, Loader2 } from 'lucide-react';

interface TimelineProps {
  status: ActionStatus;
}

export const Timeline: React.FC<TimelineProps> = ({ status }) => {
  // Determine state for each of the 5 steps
  const getStepState = (stepIndex: number) => {
    switch (stepIndex) {
      case 1: // Understand
        if (status === 'idle') return 'pending';
        return 'completed';

      case 2: // Plan / Analyze
        if (status === 'idle') return 'pending';
        if (status === 'planning' || status === 'analyzing') return 'active';
        return 'completed';

      case 3: // Approve
        if (['idle', 'planning', 'analyzing'].includes(status)) return 'pending';
        if (['unsupported', 'analysis_complete'].includes(status)) return 'skipped';
        if (status === 'pending_approval') return 'active';
        if (status === 'rejected') return 'rejected';
        return 'completed';

      case 4: // Act
        if (['idle', 'planning', 'pending_approval', 'analyzing'].includes(status)) return 'pending';
        if (['rejected', 'unsupported', 'analysis_complete'].includes(status)) return 'skipped';
        if (status === 'executing') return 'active';
        return 'completed';

      case 5: // Verify
        if (['idle', 'planning', 'pending_approval', 'executing', 'analyzing'].includes(status)) return 'pending';
        if (['rejected', 'unsupported', 'analysis_complete'].includes(status)) return 'skipped';
        if (status === 'error') return 'failed';
        return 'completed';

      default:
        return 'pending';
    }
  };

  const getStepDesc = (stepNumber: number, defaultDesc: string) => {
    if (status === 'analysis_complete' || status === 'analyzing') {
      if (stepNumber === 2) return 'Document analyzed';
      if (stepNumber === 3) return 'Not required (Read-only)';
      if (stepNumber === 4) return 'No execution';
      if (stepNumber === 5) return 'No disk changes';
    }
    return defaultDesc;
  };

  const steps = [
    { number: 1, label: 'Understand', desc: 'Intent parsed', icon: Brain },
    { number: 2, label: 'Plan', desc: 'Steps formulated', icon: FileText },
    { number: 3, label: 'Approve', desc: 'Human gate', icon: Lock },
    { number: 4, label: 'Act', desc: 'Execute safely', icon: Play },
    { number: 5, label: 'Verify', desc: 'Disk check', icon: CheckCircle },
  ];

  return (
    <div className="timeline-compact-strip" role="region" aria-label="Workflow progress">
      <div className="timeline-compact-header">
        <span className="timeline-compact-badge">KAIROS Workflow</span>
        <span className={`status-pill compact status-${status}`}>
          {status === 'pending_approval' && 'Waiting for Approval'}
          {status === 'planning' && 'Gemma Thinking...'}
          {status === 'analyzing' && 'Analyzing Document...'}
          {status === 'unsupported' && 'Answered (No file action)'}
          {status === 'executing' && 'Executing Action...'}
          {status === 'completed' && 'Completed & Verified'}
          {status === 'analysis_complete' && 'Document Analysis Ready'}
          {status === 'rejected' && 'Action Rejected'}
          {status === 'idle' && 'Awaiting Input'}
          {status === 'error' && 'Error Occurred'}
        </span>
      </div>

      <div className="timeline-compact-steps">
        {steps.map((step, idx) => {
          const state = getStepState(step.number);
          const desc = getStepDesc(step.number, step.desc);
          const Icon = step.icon;

          return (
            <React.Fragment key={step.number}>
              <div
                className={`timeline-compact-step state-${state}`}
                title={`${step.number}. ${step.label}: ${desc}`}
                tabIndex={0}
              >
                <div className="step-compact-icon">
                  {state === 'active' ? (
                    <Loader2 className="step-spinner animate-spin" size={13} />
                  ) : state === 'rejected' ? (
                    <XCircle size={13} className="text-red" />
                  ) : (
                    <Icon size={13} />
                  )}
                </div>
                <span className="step-compact-label">{step.label}</span>
              </div>
              {idx < steps.length - 1 && (
                <div className={`timeline-compact-line line-${state}`} />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
