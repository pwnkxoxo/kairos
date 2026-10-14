import React, { useState } from 'react';
import {
  ShieldCheck,
  Activity,
  Plus,
  Search,
  MessageSquare,
  Trash2,
  Edit2,
  Check,
  X,
  Cpu,
  Database,
} from 'lucide-react';
import { HealthResponse, ConversationSummary } from '../types/api';

interface SidebarProps {
  health: HealthResponse | null;
  healthLoading: boolean;
  onRefreshHealth: () => void;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onRenameConversation: (id: string, newTitle: string) => void;
  onDeleteConversation: (id: string) => void;
  disabled: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  health,
  healthLoading,
  onRefreshHealth,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onRenameConversation,
  onDeleteConversation,
  disabled,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const isHealthy = health?.status === 'healthy';
  const isGemma = health?.mode === 'gemma';

  // Filter conversations by title
  const filteredConversations = conversations.filter((c) =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase().trim())
  );

  const startEditing = (conv: ConversationSummary, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditTitle(conv.title);
  };

  const saveEditing = (id: string, e: React.MouseEvent | React.FormEvent) => {
    e.stopPropagation();
    if (editTitle.trim()) {
      onRenameConversation(id, editTitle.trim());
    }
    setEditingId(null);
  };

  const cancelEditing = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(null);
  };

  const confirmDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeletingId(id);
  };

  const executeDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    onDeleteConversation(id);
    setDeletingId(null);
  };

  const cancelDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDeletingId(null);
  };

  return (
    <aside className="sidebar">
      {/* 1. KAIROS Branding and application identity */}
      <div className="brand-container">
        <div className="brand-logo">
          <div className="pulse-ring" />
          <ShieldCheck className="brand-icon" size={24} />
        </div>
        <div>
          <h1 className="brand-title">KAIROS</h1>
          <p className="brand-tagline">Speak. Think. Execute.</p>
        </div>
      </div>

      {/* 2. Compact Backend Status & Model Indicator */}
      <div className="status-card-compact">
        <div className="status-compact-row">
          <div className="status-indicator">
            <span className={`status-dot ${isHealthy ? 'online' : 'offline'}`} />
            <span className="status-compact-text">
              {isHealthy ? `KAIROS v${health?.version || '0.4.0'}` : 'Backend Offline'}
            </span>
          </div>

          <button
            className="refresh-btn-compact"
            onClick={onRefreshHealth}
            disabled={healthLoading}
            title="Refresh status"
          >
            {healthLoading ? '...' : <Activity size={12} />}
          </button>
        </div>

        {health && (
          <div className="engine-status-compact">
            <Cpu size={11} className="engine-icon" />
            <span className="engine-name">
              {isGemma ? (health.model || 'Gemma 4') : 'Safe Demo Mode'}
            </span>
          </div>
        )}
      </div>

      {/* 3. Prominent New Chat Control */}
      <div className="new-chat-container">
        <button
          className="new-chat-btn"
          onClick={onNewChat}
          disabled={disabled}
          title="Start a fresh conversation"
        >
          <Plus size={16} />
          <span>New Chat</span>
        </button>
      </div>

      {/* 4. Search field for locally stored conversations */}
      <div className="search-box">
        <Search size={14} className="search-icon" />
        <input
          type="text"
          className="search-input"
          placeholder="Search conversations..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        {searchQuery && (
          <button
            className="search-clear-btn"
            onClick={() => setSearchQuery('')}
            title="Clear search"
          >
            <X size={12} />
          </button>
        )}
      </div>

      {/* 5. Conversation History List (Ordered by recency, independently scrollable) */}
      <div className="conversation-history-section">
        <div className="history-header">
          <span className="history-title">Conversations</span>
          <span className="history-count">{conversations.length}</span>
        </div>

        <div className="conversation-list" role="list">
          {filteredConversations.length === 0 ? (
            <div className="history-empty">
              {searchQuery ? (
                <p>No matching conversations found.</p>
              ) : (
                <p>No chats yet. Start a new conversation above.</p>
              )}
            </div>
          ) : (
            filteredConversations.map((conv) => {
              const isActive = conv.id === activeConversationId;
              const isEditing = editingId === conv.id;
              const isDeleting = deletingId === conv.id;

              return (
                <div
                  key={conv.id}
                  className={`conversation-item ${isActive ? 'active' : ''}`}
                  onClick={() => !isEditing && !isDeleting && onSelectConversation(conv.id)}
                  role="button"
                  tabIndex={0}
                >
                  <MessageSquare size={14} className="conv-icon" />

                  {isEditing ? (
                    <form
                      className="edit-title-form"
                      onSubmit={(e) => saveEditing(conv.id, e)}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <input
                        type="text"
                        className="edit-title-input"
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        autoFocus
                      />
                      <button
                        type="button"
                        className="btn-edit-action save"
                        onClick={(e) => saveEditing(conv.id, e)}
                        title="Save title"
                      >
                        <Check size={12} />
                      </button>
                      <button
                        type="button"
                        className="btn-edit-action cancel"
                        onClick={cancelEditing}
                        title="Cancel"
                      >
                        <X size={12} />
                      </button>
                    </form>
                  ) : isDeleting ? (
                    <div className="delete-confirm-box" onClick={(e) => e.stopPropagation()}>
                      <span className="delete-confirm-text">Delete chat?</span>
                      <button
                        type="button"
                        className="btn-del-action confirm"
                        onClick={(e) => executeDelete(conv.id, e)}
                        title="Confirm deletion"
                      >
                        Yes
                      </button>
                      <button
                        type="button"
                        className="btn-del-action cancel"
                        onClick={cancelDelete}
                        title="Cancel"
                      >
                        No
                      </button>
                    </div>
                  ) : (
                    <>
                      <span className="conv-title" title={conv.title}>
                        {conv.title}
                      </span>

                      <div className="conv-actions">
                        <button
                          type="button"
                          className="conv-action-btn"
                          onClick={(e) => startEditing(conv, e)}
                          title="Rename conversation"
                        >
                          <Edit2 size={12} />
                        </button>
                        <button
                          type="button"
                          className="conv-action-btn delete"
                          onClick={(e) => confirmDelete(conv.id, e)}
                          title="Delete conversation"
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* 6. Compact Settings / Local Storage info footer */}
      <div className="sidebar-footer-compact">
        <div className="storage-badge">
          <Database size={12} className="storage-icon" />
          <span>Local SQLite Storage</span>
        </div>
        <p className="privacy-note">Zero cloud memory sync • Human approval gate</p>
      </div>
    </aside>
  );
};
