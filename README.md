# Project KAIROS — Permission-Controlled AI Assistant

> **Speak. Think. Execute.**
> KAIROS is a safety-first AI assistant designed to eliminate rogue AI execution. Instead of letting an AI autonomously create files or run commands on your machine, KAIROS enforces a strict 5-stage human-in-the-loop workflow:
>
> **Understand → Plan → Approve → Act → Verify**

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Safety Architecture](#-safety-architecture)
3. [Implemented Features](#-implemented-features)
4. [Technology Stack & Project Structure](#-technology-stack--project-structure)
5. [Prerequisites](#-prerequisites)
6. [Installation & Setup](#-installation--setup)
7. [API Key & Environment Configuration](#-api-key--environment-configuration)
8. [OCR Setup & Troubleshooting](#-ocr-setup--troubleshooting)
9. [Running & Testing KAIROS](#-running--testing-kairos)
10. [Data Storage & Privacy](#-data-storage--privacy)
11. [Known Limitations & Troubleshooting](#-known-limitations--troubleshooting)
12. [Roadmap & Contribution Guidelines](#-roadmap--contribution-guidelines)

---

## 📖 Project Overview

### What Problem Does KAIROS Solve?
Modern generative AI coding tools and autonomous agents often execute actions directly against your operating system, running arbitrary scripts or writing unreviewed files. When models hallucinate or misinterpret user intent, this can lead to accidental data loss, unvetted changes, or security vulnerabilities.

KAIROS solves this by placing an **uncompromising human approval gate** between AI reasoning and operating system execution:
- The AI can formulate plans and propose structured changes.
- **Zero files are created or modified** until you review the exact diff or file preview and click **Approve**.
- If you click **Reject**, the action is immediately cancelled and discarded.
- In addition to safe planning, KAIROS features **Document Intelligence** for analyzing uploaded PDF documents (exam papers, lecture slides, syllabus documents) with local OCR fallback and grounded question answering.

> [!IMPORTANT]
> **Network Connectivity Requirement**:
> KAIROS is **not a fully offline AI application**. While SQLite conversation history, PDF parsing (`pypdf`), and OCR processing (`tesseract`) operate entirely on your local machine, **Gemma 4 planning and document Q&A require outbound internet connectivity** to Google's hosted AI APIs (`google-genai`).

---

## 🛡️ Safety Architecture

KAIROS enforces a 5-stage permission workflow:

```
[User Message / PDF] ──▶ 1. UNDERSTAND ──▶ 2. PLAN (Gemma 4 AI)
                                                  │
                                                  ▼
                                          3. APPROVE (Human Gate)
                                          ├── [Reject] ──▶ Cancelled (No disk action)
                                          └── [Approve]
                                                  │
                                                  ▼
                                            4. ACT (Safe Write in workspace/)
                                                  │
                                                  ▼
                                            5. VERIFY (Verify file size & path)
```

1. **Understand**: Parses user input, extracts PDF context, and retrieves bounded conversational history from SQLite.
2. **Plan**: Gemma 4 (`gemma-4-26b-a4b-it`) generates a structured JSON plan specifying the action and exact proposed content.
3. **Approve (Human Gate)**: The frontend displays an action card showing the exact plan and code preview. Execution halts completely until the user explicitly clicks **Approve** or **Reject**.
4. **Act**: Only approved actions are executed. File writes are restricted to `workspace/` and path traversal attempts (e.g. `../../`) are blocked. Shell execution is prohibited.
5. **Verify**: KAIROS verifies that the written file exists on disk and is non-empty before reporting success.

---

## ✨ Implemented Features

### 🟢 Fully Implemented & Verified
- **Natural-Language Chat & Safe File Creation**: Conversational interface that turns prompts into structured plans. Writes validated content to `workspace/demo.txt` upon human approval.
- **Human Approval & Rejection Gates**: Dedicated UI buttons to approve or reject pending actions. Idempotency guarantees prevent re-approving or altering already completed actions.
- **Dual Execution Modes**:
  - **Gemma 4 Mode**: Live reasoning and structured planning using hosted `gemma-4-26b-a4b-it` via the official `google-genai` SDK.
  - **Safe Demo Mode**: Deterministic offline fallback template when `GEMINI_API_KEY` is not configured.
- **PDF Upload & Text Extraction**: Accepts multipart PDF uploads up to 10 MB and 50 pages. Extracts text using `pypdf`.
- **Image Upload & OCR**: Accepts image uploads (`image/png`, `image/jpeg`, `image/webp` up to 10 MB). Extracts printed text locally using `pytesseract` and `Pillow`. Never uploads image bytes to external models.
- **Webpage URL Ingestion**: Ingests public web URLs safely using `httpx` and `readabilipy`/`BeautifulSoup4` with SSRF safeguards (private/reserved IP blocking, localhost filtering, scheme enforcement).
- **Local OCR Fallback**: Scanned PDF pages with minimal native text automatically trigger local optical character recognition via `pypdfium2` and `pytesseract` (Tesseract OCR). Page images never leave your local machine.
- **Read-Only Document, Image & Web Q&A**: Answers grounded questions against uploaded PDFs, images, or ingested webpages. Enforces strict grounding prompts to prevent hallucinated assumptions. Never triggers arbitrary file creation or system commands.
- **Markdown & LaTeX Rendering**: Full mathematical expression and markdown support using `react-markdown`, `remark-math`, `rehype-katex`, and `katex` CSS.
- **Local SQLite Conversation History**: Stores conversations, messages, and document links in an OS-appropriate data directory (`kairos_history.db`, with automatic backward compatibility for existing `echo_history.db`). Supports creating, switching, renaming, searching, and deleting chats.
- **Timezone-Aware System Clock**: Directly and accurately answers current date, time, and day questions locally in milliseconds using Python standard library and configured timezone (`KAIROS_TIMEZONE`, default: `Asia/Kolkata`) without making external API calls.
- **Live Web Research (Tavily Search API)**: Selectively searches the web for current news, live updates, and post-cutoff queries using the official `tavily-python` SDK. Bounded search evidence is synthesized by Gemma 4 with source citations while treating web content as untrusted data.
- **Source-Aware Quick Actions**: Dynamically adapts prompt suggestions based on attached context:
  - *Webpages*: Summarize webpage, Key takeaways, Explain simply, Action checklist.
  - *PDFs & Documents*: Revision Map (with page citations), Summarize, Explain simply, Revision notes, Practice quiz, Action checklist.
  - *Images*: Extract text & details, Summarize content, Key takeaways.
  - *Mixed / Multi-source*: Summarize attachments, Key takeaways across sources, Compare sources, Action checklist.
- **Action Results & Feedback**: Copy answer button with visual feedback, retry button for failed requests, and clear error banners.

### 🟡 Partial Features & Browser Limitations
- **Voice Input (Speech-to-Text)**: Uses the browser's native `webkitSpeechRecognition` / `SpeechRecognition` Web API.
  *Limitation*: Subject to browser support (best in Google Chrome/Chromium). Some Linux browsers or non-Chromium browsers may throw a network or permission error.
- **Read Aloud (Text-to-Speech)**: Uses the browser's `window.speechSynthesis` API. Formatted math and code blocks are stripped for cleaner vocal output.
  *Limitation*: Voice quality and accent support depend entirely on the host operating system's installed speech synthesis voices.

---

## 🏗️ Technology Stack & Project Structure

### Technology Stack
- **Frontend**: React 18, TypeScript, Vite 5, Lucide React, KaTeX, React Markdown.
- **Backend**: FastAPI, Uvicorn, Pydantic v2, Python 3.10+.
- **AI Integration**: Google Gen AI SDK (`google-genai`), model: `gemma-4-26b-a4b-it`.
- **Document Processing**: `pypdf` (native text), `pypdfium2` (PDF page rendering), `pytesseract` + `Pillow` (local OCR).
- **Storage**: SQLite3 with thread-safe parameterized queries and XDG data directory support.
- **Testing**: Python `unittest`, FastAPI `TestClient` (HTTPX).

### Project Directory Structure
```
kairos/
├── .env.example             # Configuration template for API keys & models
├── .gitignore               # Ignores .env, .venv, .tools, build artifacts, databases
├── README.md                # Project documentation
├── backend/
│   ├── __init__.py
│   ├── agent.py             # CLI prototype demonstrating the 5-step loop
│   ├── api.py               # FastAPI backend endpoints (Chat, Actions, PDF, Images, URLs, Conversations)
│   ├── model_client.py      # Hosted Gemma client, JSON parsing, error sanitization, timeouts
│   ├── pdf_service.py       # PDF validation, image handling, OCR (Native + Tesseract OCR)
│   ├── requirements.txt     # Python backend dependencies
│   ├── search_service.py    # Local date/time handling and Tavily web search integration
│   ├── storage.py           # SQLite local conversation history and document persistence
│   └── url_service.py       # Safe URL ingestion, SSRF protection, readability extraction
├── frontend/
│   ├── index.html           # Single-page application entry point
│   ├── package.json         # Frontend scripts and dependencies
│   ├── tsconfig.json        # TypeScript configuration
│   ├── vite.config.ts       # Vite bundler configuration with backend proxy
│   └── src/
│       ├── App.tsx          # Main application canvas, state orchestration, audio handlers
│       ├── main.tsx         # React root mounting
│       ├── index.css        # Dark theme stylesheet and responsive layout
│       ├── components/
│       │   ├── ChatInput.tsx         # Prompt composer, attachments, source-aware quick actions, microphone
│       │   ├── ExecutionResult.tsx   # Verified execution/rejection feedback card
│       │   ├── MarkdownRenderer.tsx  # React-Markdown with KaTeX LaTeX math support
│       │   ├── PlanCard.tsx          # Human approval gate card with diff/content preview
│       │   ├── Sidebar.tsx           # Conversation list, search, health indicator, new chat
│       │   └── Timeline.tsx          # Visual 5-stage progress indicator
│       ├── services/
│       │   └── api.ts       # Frontend HTTP client with timeout and abort handling
│       └── types/
│           └── api.ts       # TypeScript interfaces for API contracts and data models
├── tests/
│   ├── __init__.py
│   ├── test_agent.py        # Tests for CLI agent sandbox and safety
│   ├── test_api.py          # Tests for FastAPI endpoints, approval, rejection, and overwrite confirmation
│   ├── test_image.py        # Tests for image validation, OCR processing, and prompt safety
│   ├── test_model_client.py # Tests for model parsing, validation, timeouts, and error hygiene
│   ├── test_pdf.py          # Tests for PDF upload, page limits, OCR fallback, and prompt safety
│   ├── test_search.py       # Tests for local date/time, timezone, and Tavily search integration
│   ├── test_storage.py      # Tests for SQLite conversations, context slicing, and cleanup
│   └── test_url.py          # Tests for URL ingestion, SSRF prevention, and content extraction
└── workspace/               # Isolated directory for approved file actions
    ├── .gitkeep
    └── pdfs/                # Local directory for stored PDF documents (<uuid>.pdf)
        └── .gitkeep
```

---

## 📋 Prerequisites

Before installing KAIROS, verify that the following tools are installed on your machine:

1. **Git**: Version control system.
   ```bash
   git --version
   ```
2. **Python**: Supported versions are **Python 3.10 to Python 3.14**.
   ```bash
   python3 --version
   ```
3. **Node.js & npm**: Node.js **v18.0.0+** (v20+ or v22+ recommended) and npm **v9.0.0+**.
   ```bash
   node --version
   npm --version
   ```
4. **Tesseract OCR (Required for Scanned PDFs)**:
   - The Python library `pytesseract` is a wrapper; it requires the underlying **Tesseract OCR system executable** and English trained data (`eng`).
   - Check if Tesseract is installed:
     ```bash
     tesseract --version
     tesseract --list-langs
     ```

---

## 💻 Installation & Setup

### Fedora / Red Hat Linux

1. **Install system packages and Tesseract OCR**:
   ```bash
   sudo dnf install -y python3 python3-pip git tesseract tesseract-langpack-eng
   ```
2. **Clone the repository**:
   ```bash
   git clone https://github.com/pwnkxoxo/kairos.git
   cd kairos
   ```
3. **Set up Python virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. **Install backend dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```
5. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```
6. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env using nano or your preferred editor to add your API key:
   # GEMINI_API_KEY=your_actual_gemini_api_key
   ```
7. **Start the FastAPI backend** (Terminal 1):
   ```bash
   .venv/bin/uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
   ```
8. **Start the Vite frontend** (Terminal 2):
   ```bash
   cd frontend
   npm run dev
   ```
9. **Open KAIROS**:
   Navigate to `http://localhost:5173` in your browser. Verify backend connectivity at `http://127.0.0.1:8000/health`.

---

### Windows (PowerShell)

1. **Install Prerequisites**:
   - Install Python 3.10+ from [python.org](https://www.python.org/) (ensure "Add Python to PATH" is checked).
   - Install Node.js LTS from [nodejs.org](https://nodejs.org/).
   - Install Tesseract OCR using `winget`:
     ```powershell
     winget install UB-Mannheim.TesseractOCR
     ```
     Ensure `C:\Program Files\Tesseract-OCR` is added to your System `PATH`.
2. **Clone and enter repository**:
   ```powershell
   git clone https://github.com/pwnkxoxo/kairos.git
   cd kairos
   ```
3. **Create and activate virtual environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
4. **Install backend dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r backend\requirements.txt
   ```
5. **Install frontend dependencies**:
   ```powershell
   cd frontend
   npm install
   cd ..
   ```
6. **Configure environment**:
   ```powershell
   Copy-Item .env.example .env
   # Open .env in Notepad to add your GEMINI_API_KEY
   notepad .env
   ```
7. **Start backend** (Terminal 1):
   ```powershell
   .venv\Scripts\uvicorn.exe backend.api:app --host 127.0.0.1 --port 8000 --reload
   ```
8. **Start frontend** (Terminal 2):
   ```powershell
   cd frontend
   npm run dev
   ```
9. **Access**:
   Open `http://localhost:5173` in your browser.

---

## 🔑 API Key & Environment Configuration

KAIROS uses environment variables defined in a local `.env` file at the root of the project.

### Configuration Template (`.env`)
```env
# Google Gemini / Gemma API Key from Google AI Studio
# Obtain from: https://aistudio.google.com/
GEMINI_API_KEY=your_actual_api_key_here

# Hosted Gemma Model Name (default: gemma-4-26b-a4b-it)
GEMMA_MODEL=gemma-4-26b-a4b-it

# Model Request Timeout in Milliseconds (default: 60000 = 60 seconds)
GEMINI_REQUEST_TIMEOUT_MS=60000

# Optional: Custom SQLite application data directory override
# KAIROS_DATA_DIR=/path/to/custom/data/directory
# Legacy fallback also recognized: ECHO_DATA_DIR
```

### Safety and Privacy Rules
> [!CAUTION]
> **NEVER commit your `.env` file to Git.**
> The `.env` file contains sensitive API keys. Verify that it is properly ignored by Git:
> ```bash
> git check-ignore -v .env
> # Expected output: .gitignore:153:.env    .env
> ```

---

## 🔍 OCR Setup & Troubleshooting

When an uploaded PDF contains scanned pages or images without embedded selectable text, KAIROS uses local OCR to extract characters.

### Installing Tesseract System Packages
- **Fedora**:
  ```bash
  sudo dnf install -y tesseract tesseract-langpack-eng
  ```
- **Ubuntu / Debian**:
  ```bash
  sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
  ```
- **Windows**:
  ```powershell
  winget install UB-Mannheim.TesseractOCR
  ```
  *(Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`)*

### Technical Constraints & Fallbacks
- **Native Text First**: If a page has 40+ native characters (`MIN_NATIVE_CHARS = 40`), native extraction via `pypdf` is used and OCR is skipped for speed.
- **Document & Page Limits**:
  - Maximum upload size: **10 MB** (`MAX_PDF_BYTES`).
  - Maximum page count: **50 pages** (`MAX_PDF_PAGES`).
  - Per-page OCR timeout: **15 seconds** (`PAGE_OCR_TIMEOUT_SECONDS`).
  - Total document OCR timeout: **60 seconds** (`DOC_OCR_TIMEOUT_SECONDS`).
- **If Tesseract is Missing**: KAIROS does not crash; it extracts whatever native text is available and alerts the user with an `OCR Missing` badge and platform installation guidance.
- **Mathematical Formula Limitations**: OCR engines can misread complex mathematical symbols, superscripts, subscripts, or fractions. If extracted formulas are ambiguous, Gemma is instructed to highlight assumptions and request clarification rather than fabricating equations.

---

## 🧪 Running & Testing KAIROS

### Running Backend Automated Tests
KAIROS features a comprehensive offline test suite covering security boundaries, approval logic, PDF extraction, SQLite persistence, and error sanitization. Automated tests execute with zero external network calls (model calls are fully mocked):

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### Building the Frontend for Production
To verify TypeScript compilation and produce the optimized Vite production bundle:

```bash
cd frontend
npm run build
cd ..
```

### Manual Smoke-Test Checklist
To verify a live running KAIROS instance:
1. **Health Endpoint**: Visit `http://127.0.0.1:8000/health`. Confirm `status: "healthy"` and mode shows `gemma` (with API key) or `demo` (without key).
2. **Conversations**: Click **+ New Chat** in the sidebar. Verify the new conversation appears in the sidebar list.
3. **Approval Gate (Normal Chat)**:
   - Type `Create a project notes file with 3 action items`.
   - Ensure KAIROS transitions to `Gemma is thinking...` and presents a **Pending Approval** plan card with a preview of `workspace/demo.txt`.
   - Click **Approve**. Confirm the file is written to `workspace/demo.txt` and verified.
4. **Rejection Gate**:
   - Send another file creation request.
   - Click **Reject**. Confirm the status changes to **Action Rejected** and no file changes occur.
5. **PDF Upload & Q&A**:
   - Click the **+** button in the composer and attach a PDF document.
   - Click one of the quick actions (e.g. *Summarize* or *Revision notes*).
   - Verify the analysis card displays formatted Markdown and LaTeX with page citations.
6. **Voice Input & Read Aloud**:
   - Click the microphone icon to test voice transcription (in Chrome).
   - Click **Read Aloud** on an assistant response to test speech synthesis.

---

## 🔒 Data Storage & Privacy

### Where Is Data Stored?
- **SQLite Database (`kairos_history.db` / `echo_history.db`)**:
  Stored locally in the user's standard application data directory:
  - **Linux**: `~/.local/share/kairos/` (or `$XDG_DATA_HOME/kairos/`)
  - **Windows**: `%APPDATA%\kairos\`
  - **macOS**: `~/Library/Application Support/kairos/`
  - *Override*: Configurable via the `KAIROS_DATA_DIR` or `ECHO_DATA_DIR` environment variable. Defaults to backward-compatible database file resolution.
- **Uploaded PDFs**:
  Stored in `workspace/pdfs/<uuid4>.pdf` using server-generated UUIDs. User filenames are stored only as display metadata in SQLite.
- **Conversation Deletion**:
  Deleting a conversation via the sidebar removes its messages from SQLite and deletes any linked PDF files that are no longer referenced by any other conversation.

### Hosted Model & Privacy Truth
- **Prompts & Context**: When `GEMINI_API_KEY` is configured, user prompts and extracted document text are transmitted to Google's Gen AI endpoints.
- **Local OCR**: Scanned document rendering and OCR processing are **100% local**. Raw page images are never uploaded to the model; only the resulting plain text context is sent.
- **Browser Speech APIs**: Voice transcription uses the browser's speech recognition engine, which may transmit audio to your browser vendor's servers for processing.

---

## ⚠️ Known Limitations & Troubleshooting

### Known Limitations
1. **Session-Scoped Document Registry**: The active mapping of in-memory PDF context is session-scoped. If the backend server restarts while a browser tab remains open, re-analyzing an existing attached document will prompt a re-upload.
2. **Speech Recognition Browser Dependencies**: Non-Chromium browsers (Firefox, Safari) or Linux environments without speech dispatcher libraries may report a `Voice input error: network` or `Speech recognition not supported`.
3. **Mathematical OCR Degradation**: Hand-written or low-resolution math formulas in scanned PDFs may yield garbled characters.
4. **Duplicate Messages on Strict Retries**: Rapidly double-clicking send during network lag can occasionally spawn parallel turns if not debounced.

### Troubleshooting Common Errors
- **`502 Bad Gateway / ModelClientError`**: Check that your `GEMINI_API_KEY` is valid and has sufficient quota in Google AI Studio.
- **`Request timed out`**: Ensure `GEMINI_REQUEST_TIMEOUT_MS` in `.env` is set to `60000` (milliseconds) and not an integer in seconds.
- **`Tesseract not found`**: Verify that the Tesseract executable is installed and available in your system `PATH`.
- **`Port 8000 already in use`**: Terminate any lingering background Uvicorn processes:
  ```bash
  fuser -k 8000/tcp
  ```

---

## 🗺️ Roadmap & Contribution Guidelines

### Implemented vs. Planned
| Feature | Status |
| :--- | :--- |
| 5-Stage Permission Gate (Understand → Plan → Approve → Act → Verify) | ✅ Implemented |
| Hosted Gemma 4 Integration (`gemma-4-26b-a4b-it`) | ✅ Implemented |
| Safe Offline Demo Mode | ✅ Implemented |
| PDF Upload & `pypdf` Text Extraction | ✅ Implemented |
| Local Tesseract OCR Fallback & Mixed Documents | ✅ Implemented |
| Markdown & KaTeX LaTeX Math Rendering | ✅ Implemented |
| SQLite Local History (Persistence across restarts) | ✅ Implemented |
| Browser Voice Input & Read Aloud | 🟡 Working (Browser dependent) |
| Multi-file workspace modifications | ⏳ Planned |
| Persistent document embeddings / Vector DB | ⏳ Planned |
| Fully offline local LLM execution (via Ollama / llama.cpp) | ⏳ Planned |

### Contribution Guidelines
1. Always create a feature branch (`git checkout -b feature/your-feature`) rather than committing directly to `main`.
2. Ensure all unit tests pass before submitting changes:
   ```bash
   .venv/bin/python -m unittest discover -s tests -v
   ```
3. Ensure the frontend builds cleanly without TypeScript errors:
   ```bash
   cd frontend && npm run build
   ```
4. Never disable or bypass the human approval gate in `backend/api.py`.

---

## 📄 License
This project is developed for hackathons and educational research in safe AI agent systems.
