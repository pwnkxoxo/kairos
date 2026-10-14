"""
KAIROS PDF Service
Handles PDF upload validation, safe storage, hybrid text extraction (Native + Local OCR),
and context preparation for Gemma AI.

Design & Security Principles:
  - All files are stored strictly inside workspace/pdfs/ using server-generated UUID4 filenames.
  - User-supplied filenames are sanitized and stored as display metadata only; never in filesystem paths.
  - Hybrid extraction: Fast native text first via pypdf. Pages with minimal or no native text
    fallback to local OCR using pypdfium2 page rendering + pytesseract (Tesseract OCR).
  - OCR is 100% local; page images are never transmitted externally.
  - Strict timeouts per page (15s) and per document (60s) to prevent hangs.
  - Detects Tesseract availability and provides platform-specific installation instructions if missing.
  - Source tracking: Every page records its extraction method ('native', 'ocr', 'ocr_empty', etc.).
  - Extracted text is sanitized (null bytes and control characters stripped) and context is bounded.
  - Session-scoped registry with graceful degradation.
"""

from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, List, Optional
import uuid

from PIL import Image, ImageOps
import pypdfium2 as pdfium
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import pytesseract

# ------------------------------------------------------------------------------
# 1. Storage Configuration & Resource Limits
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_STORAGE_DIR = PROJECT_ROOT / "workspace" / "pdfs"

MAX_PDF_BYTES = 10 * 1024 * 1024    # 10 MB limit for PDFs
MAX_IMAGE_BYTES = 5 * 1024 * 1024   # 5 MB limit for Images
MAX_PDF_PAGES = 50
MAX_CONTEXT_CHARS = 8_000

# Image Security Limits (Decompression Bomb Protection)
MAX_IMAGE_PIXELS = 25_000_000       # 25 Megapixels (e.g. 5000x5000)
MAX_IMAGE_DIMENSION = 6000          # Max width or height

# Supported Image Formats
SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}

# OCR Tuning Parameters
MIN_NATIVE_CHARS = 40             # Minimum characters for a page to be considered native text
OCR_DPI = 200                     # Default rendering DPI (optimal balance of OCR accuracy and processing speed)
PAGE_OCR_TIMEOUT_SECONDS = 20     # Timeout for OCR on a single page
DOC_OCR_TIMEOUT_SECONDS = 60      # Cumulative timeout for OCR processing on an entire document
DEFAULT_OCR_LANG = "eng"          # Default language pack
MAX_OCR_PROCESS_DIMENSION = 2500  # Clamp pixel dimension for fast, reliable OCR without OOM or timeouts

# Configure Pillow decompression bomb ceiling
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# ------------------------------------------------------------------------------
# 2. In-Memory Document Registry (Session-Scoped)
# ------------------------------------------------------------------------------
pdf_registry: Dict[str, Dict[str, Any]] = {}


def clear_pdf_registry() -> None:
    """Helper used primarily by unit tests to clean up document state."""
    pdf_registry.clear()


# ------------------------------------------------------------------------------
# 3. Path Security & Sanitization
# ------------------------------------------------------------------------------
def _resolve_storage_path(doc_id: str, extension: str = "pdf") -> Path:
    """
    Validates that doc_id is a valid UUID4 and resolves its absolute path
    strictly within PDF_STORAGE_DIR with a sanitized extension (.pdf, .png, .jpg, .webp, .txt).
    Prevents path traversal attacks (e.g. '../../etc/passwd').
    """
    try:
        parsed_uuid = uuid.UUID(doc_id, version=4)
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"Invalid document ID format: {doc_id!r}")

    safe_ext = extension.lower().lstrip(".")
    if safe_ext not in {"pdf", "png", "jpg", "jpeg", "webp", "txt"}:
        safe_ext = "pdf"
    if safe_ext == "jpeg":
        safe_ext = "jpg"

    target = (PDF_STORAGE_DIR / f"{parsed_uuid}.{safe_ext}").resolve()
    if not target.is_relative_to(PDF_STORAGE_DIR.resolve()):
        raise PermissionError(f"Security Alert: Document path '{target}' escapes storage directory!")
    return target


def _find_storage_path(doc_id: str) -> Path:
    """
    Finds the existing storage path for doc_id across supported extensions (.pdf, .png, .jpg, .webp, .txt).
    If none exists on disk, returns the default .pdf path.
    """
    try:
        parsed_uuid = uuid.UUID(doc_id, version=4)
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"Invalid document ID format: {doc_id!r}")

    for ext in ["pdf", "png", "jpg", "webp", "txt"]:
        candidate = (PDF_STORAGE_DIR / f"{parsed_uuid}.{ext}").resolve()
        if candidate.exists() and candidate.is_relative_to(PDF_STORAGE_DIR.resolve()):
            return candidate
    return (PDF_STORAGE_DIR / f"{parsed_uuid}.pdf").resolve()


def _sanitize_display_filename(filename: str) -> str:
    """Extracts a safe basename for UI display, stripping directory characters."""
    clean_name = Path(filename).name
    clean_name = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", clean_name).strip()
    return clean_name[:255] if clean_name else "document"


def detect_file_type(file_bytes: bytes) -> str:
    """
    Detects actual file signature (magic bytes) independent of user-supplied filenames or MIME headers.
    Returns: 'application/pdf', 'image/png', 'image/jpeg', 'image/webp', or 'unknown'.
    """
    if len(file_bytes) < 4:
        return "unknown"

    if file_bytes.startswith(b"%PDF"):
        return "application/pdf"
    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if file_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if file_bytes[:4] == b"RIFF" and len(file_bytes) >= 12 and file_bytes[8:12] == b"WEBP":
        return "image/webp"

    return "unknown"


def _clean_extracted_text(text: str) -> str:
    """
    Cleans extracted PDF/OCR text:
    - Removes null bytes (\\x00) and dangerous control characters that cause upstream API errors.
    - Normalizes excessive whitespace while preserving newlines and indentation.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return cleaned.strip()


# ------------------------------------------------------------------------------
# 4. Tesseract OCR Detection & Platform Guidance
# ------------------------------------------------------------------------------
def is_tesseract_available() -> bool:
    """Checks whether the Tesseract OCR executable is available in PATH or configured."""
    return shutil.which("tesseract") is not None or shutil.which(pytesseract.pytesseract.tesseract_cmd) is not None


def get_tesseract_install_instructions() -> str:
    """Returns platform-specific instructions for installing Tesseract."""
    return (
        "Tesseract OCR is not installed. To enable OCR on Fedora: 'sudo dnf install tesseract tesseract-langpack-eng' | "
        "Ubuntu/Debian: 'sudo apt-get install tesseract-ocr tesseract-ocr-eng' | "
        "Windows: download from https://github.com/UB-Mannheim/tesseract/wiki"
    )


# ------------------------------------------------------------------------------
# 5. Local OCR Rendering & Processing Helpers
# ------------------------------------------------------------------------------
def render_page_to_image(storage_path: Path, page_index: int, dpi: int = OCR_DPI) -> Image.Image:
    """
    Renders an individual PDF page to a PIL Image at the specified DPI using pypdfium2.
    Safe memory management: Clamps excessively large dimensions.
    """
    pdf = pdfium.PdfDocument(str(storage_path))
    try:
        if page_index < 0 or page_index >= len(pdf):
            raise IndexError(f"Page index {page_index} out of range (document has {len(pdf)} pages)")
        page = pdf[page_index]
        scale = dpi / 72.0
        bitmap = page.render(scale=scale)
        return bitmap.to_pil()
    finally:
        pdf.close()


def run_page_ocr(
    image: Image.Image,
    lang: str = DEFAULT_OCR_LANG,
    timeout: int = PAGE_OCR_TIMEOUT_SECONDS,
) -> str:
    """
    Executes local Tesseract OCR on a PIL image with orientation awareness and timeout:
      1. Clamps image dimension to MAX_OCR_PROCESS_DIMENSION to avoid CPU/memory blowup.
      2. Runs a fast thumbnail orientation score to auto-detect inverted or rotated scans.
      3. Performs full OCR on the optimally oriented image.
    Raises TesseractNotFoundError, TesseractError, or TimeoutError on failure.
    """
    proc_img = image
    if max(image.size) > MAX_OCR_PROCESS_DIMENSION:
        scale = MAX_OCR_PROCESS_DIMENSION / max(image.size)
        proc_img = image.resize((int(image.width * scale), int(image.height * scale)))

    # Fast orientation check using downsampled thumbnail (~600px max)
    best_angle = 0
    try:
        thumb_scale = 600.0 / max(proc_img.size)
        thumb = proc_img.resize((int(proc_img.width * thumb_scale), int(proc_img.height * thumb_scale)))

        # Baseline at 0 deg
        data0 = pytesseract.image_to_data(thumb, lang=lang, output_type=pytesseract.Output.DICT)
        words0 = [w for w, c in zip(data0["text"], data0["conf"]) if int(c) > 30 and len(w.strip()) > 1]
        confs0 = [int(c) for c in data0["conf"] if int(c) > 0]
        avg0 = (sum(confs0) / len(confs0)) if confs0 else 0
        score0 = len(words0) * (avg0 / 100.0)

        # If baseline orientation has poor confidence/word density, check 180, 90, 270
        if score0 < 45:
            best_score = score0
            for angle in [180, 90, 270]:
                r_thumb = thumb.rotate(angle, expand=True)
                data_r = pytesseract.image_to_data(r_thumb, lang=lang, output_type=pytesseract.Output.DICT)
                words_r = [w for w, c in zip(data_r["text"], data_r["conf"]) if int(c) > 30 and len(w.strip()) > 1]
                confs_r = [int(c) for c in data_r["conf"] if int(c) > 0]
                avg_r = (sum(confs_r) / len(confs_r)) if confs_r else 0
                score_r = len(words_r) * (avg_r / 100.0)
                if score_r > best_score:
                    best_score = score_r
                    best_angle = angle
    except Exception:
        # If thumbnail sampling fails or throws, fallback safely to standard 0-deg orientation
        best_angle = 0

    final_img = proc_img.rotate(best_angle, expand=True) if best_angle else proc_img
    return pytesseract.image_to_string(final_img, lang=lang, timeout=timeout)


# ------------------------------------------------------------------------------
# 6. Upload Validation & Safe Storage
# ------------------------------------------------------------------------------
def validate_and_store_pdf(file_bytes: bytes, original_filename: str) -> Dict[str, Any]:
    """
    Validates uploaded bytes and securely saves the PDF:
      1. Checks PDF magic header (%PDF).
      2. Checks size limit (<= 10 MB).
      3. Parses structure with pypdf to check encryption and corruption.
      4. Checks page limit (<= 50 pages).
      5. Generates UUID4 and writes to workspace/pdfs/<uuid>.pdf.

    Raises ValueError with a user-friendly error message on any validation failure.
    No file is saved if validation fails.
    """
    # 1. Magic bytes check
    if not file_bytes.startswith(b"%PDF"):
        raise ValueError("Uploaded file is not a valid PDF (missing %PDF header).")

    # 2. File size check
    size_bytes = len(file_bytes)
    if size_bytes > MAX_PDF_BYTES:
        max_mb = MAX_PDF_BYTES // (1024 * 1024)
        raise ValueError(f"PDF exceeds the maximum allowed size of {max_mb} MB ({size_bytes} bytes uploaded).")

    # 3. Structure & encryption check via pypdf
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            raise ValueError("Uploaded PDF is encrypted. Please provide an unencrypted document.")
        page_count = len(reader.pages)
    except ValueError:
        raise
    except (PdfReadError, Exception) as exc:
        raise ValueError(f"PDF could not be parsed or is corrupted: {str(exc)[:150]}")

    # 4. Page count check
    if page_count > MAX_PDF_PAGES:
        raise ValueError(f"PDF has {page_count} pages, which exceeds the limit of {MAX_PDF_PAGES} pages.")

    if page_count == 0:
        raise ValueError("Uploaded PDF has no pages.")

    # 5. Safe write
    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    doc_id = str(uuid.uuid4())
    storage_path = _resolve_storage_path(doc_id)
    storage_path.write_bytes(file_bytes)

    sha256_prefix = hashlib.sha256(file_bytes).hexdigest()[:12]
    safe_name = _sanitize_display_filename(original_filename)

    metadata: Dict[str, Any] = {
        "doc_id": doc_id,
        "filename": safe_name,
        "page_count": page_count,
        "size_bytes": size_bytes,
        "sha256_prefix": sha256_prefix,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "storage_path": str(storage_path),
        "file_type": "pdf",
        "ocr_available": is_tesseract_available(),
    }

    pdf_registry[doc_id] = metadata
    return metadata


def validate_and_store_image(file_bytes: bytes, original_filename: str) -> Dict[str, Any]:
    """
    Validates uploaded image bytes and securely saves the file:
      1. Enforces 5 MB file size limit.
      2. Validates magic bytes / signature (PNG, JPEG, WebP).
      3. Validates with Pillow to guard against decompression bombs and corrupted streams.
      4. Validates dimension limits (width, height <= 6000px, pixels <= 25,000,000).
      5. Generates UUID4 and writes to workspace/pdfs/<uuid>.<ext>.

    Raises ValueError with a user-friendly error message on any validation failure.
    No file is saved if validation fails.
    """
    size_bytes = len(file_bytes)
    if size_bytes > MAX_IMAGE_BYTES:
        max_mb = MAX_IMAGE_BYTES // (1024 * 1024)
        raise ValueError(f"Image exceeds the maximum allowed size of {max_mb} MB ({size_bytes} bytes uploaded).")

    detected_type = detect_file_type(file_bytes)
    if detected_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(
            "Unsupported or unrecognized image format. KAIROS accepts PNG, JPEG, and WebP images."
        )

    # Determine extension
    ext_map = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
    }
    safe_ext = ext_map.get(detected_type, "png")

    # Pillow decompression bomb protection & stream verification
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                raise ValueError("Image has invalid or empty dimensions.")
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                raise ValueError(
                    f"Image dimensions ({width}x{height}) exceed maximum allowed dimension of {MAX_IMAGE_DIMENSION}px."
                )
            if (width * height) > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image total pixels ({width * height}) exceed maximum allowed ceiling of {MAX_IMAGE_PIXELS} pixels."
                )
            img.load()
    except Image.DecompressionBombError:
        raise ValueError("Image exceeds maximum pixel dimensions (decompression bomb protection).")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Image file is corrupted or could not be decoded: {str(exc)[:150]}")

    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    doc_id = str(uuid.uuid4())
    storage_path = _resolve_storage_path(doc_id, extension=safe_ext)
    storage_path.write_bytes(file_bytes)

    sha256_prefix = hashlib.sha256(file_bytes).hexdigest()[:12]
    safe_name = _sanitize_display_filename(original_filename)

    metadata: Dict[str, Any] = {
        "doc_id": doc_id,
        "filename": safe_name,
        "page_count": 1,
        "size_bytes": size_bytes,
        "sha256_prefix": sha256_prefix,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "storage_path": str(storage_path),
        "file_type": "image",
        "ocr_available": is_tesseract_available(),
    }

    pdf_registry[doc_id] = metadata
    return metadata


# ------------------------------------------------------------------------------
# 7. Hybrid Text Extraction (Native + Local OCR Fallback)
# ------------------------------------------------------------------------------
def _extract_image_text(storage_path: Path, ocr_lang: str = DEFAULT_OCR_LANG) -> List[Dict[str, Any]]:
    """
    Extracts text from an image file using local Tesseract OCR:
      - Validates Tesseract availability.
      - Loads image safely with Pillow, applies EXIF transpose for orientation.
      - Converts to RGB/L if needed.
      - Enforces PAGE_OCR_TIMEOUT_SECONDS timeout.
      - Tracks extraction method ('ocr', 'ocr_empty', 'scanned_missing_tesseract', 'ocr_missing_lang', 'ocr_error').
    """
    tesseract_available = is_tesseract_available()
    if not tesseract_available:
        missing_msg = (
            "[Image: Scanned image with no native text. Local Tesseract OCR is not installed on the server. "
            f"{get_tesseract_install_instructions()}]"
        )
        return [{
            "page": 1,
            "text": missing_msg,
            "has_text": False,
            "method": "scanned_missing_tesseract",
        }]

    try:
        with Image.open(storage_path) as img:
            transposed = ImageOps.exif_transpose(img)
            if transposed.mode not in ("RGB", "L"):
                transposed = transposed.convert("RGB")
            raw_ocr = run_page_ocr(transposed, lang=ocr_lang, timeout=PAGE_OCR_TIMEOUT_SECONDS)
            cleaned_ocr = _clean_extracted_text(raw_ocr)

            if len(cleaned_ocr) > 0:
                return [{
                    "page": 1,
                    "text": cleaned_ocr,
                    "has_text": True,
                    "method": "ocr",
                }]
            else:
                return [{
                    "page": 1,
                    "text": "[Image: No readable text detected via local OCR]",
                    "has_text": False,
                    "method": "ocr_empty",
                }]
    except pytesseract.TesseractNotFoundError:
        return [{
            "page": 1,
            "text": f"[Image: Tesseract OCR executable was not found. {get_tesseract_install_instructions()}]",
            "has_text": False,
            "method": "scanned_missing_tesseract",
        }]
    except pytesseract.TesseractError as terr:
        err_msg = str(terr).lower()
        if "language" in err_msg or "traineddata" in err_msg:
            return [{
                "page": 1,
                "text": f"[Image: Tesseract language pack '{ocr_lang}' is missing. Install with: sudo dnf install tesseract-langpack-{ocr_lang}]",
                "has_text": False,
                "method": "ocr_missing_lang",
            }]
        return [{
            "page": 1,
            "text": f"[Image: Tesseract error: {str(terr)[:100]}]",
            "has_text": False,
            "method": "ocr_error",
        }]
    except (TimeoutError, Exception) as exc:
        return [{
            "page": 1,
            "text": f"[Image: OCR processing failed or timed out: {str(exc)[:100]}]",
            "has_text": False,
            "method": "ocr_error",
        }]


def extract_text_by_page(doc_id: str, ocr_lang: str = DEFAULT_OCR_LANG) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from the stored document:
      - If image: performs orientation-corrected local OCR extraction.
      - If PDF: performs hybrid extraction (native text first via pypdf, local OCR fallback for scanned pages).
      - Enforces per-page and total-document timeouts.
      - Tracks extraction method ('native', 'ocr', 'ocr_empty', 'ocr_skipped', 'scanned_missing_tesseract').
    """
    meta = get_doc_metadata(doc_id)
    if not meta:
        raise KeyError(f"Document '{doc_id}' not found in registry. It may have expired or was deleted.")

    storage_path = Path(meta["storage_path"])
    if not storage_path.exists():
        raise FileNotFoundError(f"Stored file for document '{doc_id}' is missing from disk.")

    is_url = meta.get("file_type") == "url" or storage_path.suffix.lower() == ".txt"
    if is_url:
        try:
            content = storage_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise ValueError(f"Failed to read stored web content '{doc_id}': {str(exc)[:150]}")
        cleaned = _clean_extracted_text(content)
        return [{
            "page": 1,
            "text": cleaned,
            "has_text": bool(cleaned.strip()),
            "method": "web_scrape",
        }]

    is_image = meta.get("file_type") == "image" or storage_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    if is_image:
        return _extract_image_text(storage_path, ocr_lang=ocr_lang)

    try:
        reader = PdfReader(str(storage_path))
    except Exception as exc:
        raise ValueError(f"Failed to read stored PDF '{doc_id}': {str(exc)[:150]}")

    tesseract_available = is_tesseract_available()
    doc_start_time = time.time()
    pages: List[Dict[str, Any]] = []

    for idx, page in enumerate(reader.pages, start=1):
        # Step A: Native extraction
        try:
            raw_native = page.extract_text() or ""
        except Exception:
            raw_native = ""

        cleaned_native = _clean_extracted_text(raw_native)

        # Step B: If page has sufficient native text, keep it and skip OCR
        if len(cleaned_native) >= MIN_NATIVE_CHARS:
            pages.append({
                "page": idx,
                "text": cleaned_native,
                "has_text": True,
                "method": "native",
            })
            continue

        # Step C: Page has little or no native text -> Local OCR fallback
        # Check cumulative document OCR timeout
        if time.time() - doc_start_time > DOC_OCR_TIMEOUT_SECONDS:
            fallback_text = (
                f"[Page {idx}: OCR skipped — cumulative document processing timeout exceeded ({DOC_OCR_TIMEOUT_SECONDS}s)]"
            )
            pages.append({
                "page": idx,
                "text": fallback_text,
                "has_text": False,
                "method": "ocr_timeout",
            })
            continue

        # Check if Tesseract engine is installed locally
        if not tesseract_available:
            missing_msg = (
                f"[Page {idx}: Scanned page with no native text. Local Tesseract OCR is not installed on the server. "
                f"{get_tesseract_install_instructions()}]"
            )
            pages.append({
                "page": idx,
                "text": missing_msg,
                "has_text": False,
                "method": "scanned_missing_tesseract",
            })
            continue

        # Perform local OCR
        try:
            pil_img = render_page_to_image(storage_path, idx - 1, dpi=OCR_DPI)
            raw_ocr = run_page_ocr(pil_img, lang=ocr_lang, timeout=PAGE_OCR_TIMEOUT_SECONDS)
            cleaned_ocr = _clean_extracted_text(raw_ocr)

            if len(cleaned_ocr) > 0:
                pages.append({
                    "page": idx,
                    "text": cleaned_ocr,
                    "has_text": True,
                    "method": "ocr",
                })
            else:
                pages.append({
                    "page": idx,
                    "text": f"[Page {idx}: No usable text recovered via local OCR]",
                    "has_text": False,
                    "method": "ocr_empty",
                })
        except pytesseract.TesseractNotFoundError:
            pages.append({
                "page": idx,
                "text": f"[Page {idx}: Tesseract OCR executable was not found. {get_tesseract_install_instructions()}]",
                "has_text": False,
                "method": "scanned_missing_tesseract",
            })
        except pytesseract.TesseractError as terr:
            err_msg = str(terr).lower()
            if "language" in err_msg or "traineddata" in err_msg:
                pages.append({
                    "page": idx,
                    "text": f"[Page {idx}: Tesseract language pack '{ocr_lang}' is missing. Install with: sudo dnf install tesseract-langpack-{ocr_lang}]",
                    "has_text": False,
                    "method": "ocr_missing_lang",
                })
            else:
                pages.append({
                    "page": idx,
                    "text": f"[Page {idx}: Tesseract error: {str(terr)[:100]}]",
                    "has_text": False,
                    "method": "ocr_error",
                })
        except (TimeoutError, Exception) as exc:
            pages.append({
                "page": idx,
                "text": f"[Page {idx}: OCR processing failed or timed out: {str(exc)[:100]}]",
                "has_text": False,
                "method": "ocr_error",
            })

    return pages


# ------------------------------------------------------------------------------
# 8. Model Context Preparation
# ------------------------------------------------------------------------------
def prepare_model_context(doc_id: str, max_chars: int = MAX_CONTEXT_CHARS) -> Dict[str, Any]:
    """
    Builds bounded, page-aware text context for Gemma analysis up to max_chars.
    Preserves page numbers, source methods, and detects scanned/mixed/image status.
    """
    meta = get_doc_metadata(doc_id)
    is_image = False
    is_url = False
    if meta:
        storage_path = Path(meta.get("storage_path", ""))
        is_url = meta.get("file_type") == "url" or storage_path.suffix.lower() == ".txt"
        is_image = meta.get("file_type") == "image" or storage_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}

    pages = extract_text_by_page(doc_id)
    total_pages = len(pages)
    native_pages = sum(1 for p in pages if p["method"] in ("native", "web_scrape"))
    ocr_pages = sum(1 for p in pages if p["method"] == "ocr")
    empty_pages = sum(1 for p in pages if not p["has_text"])

    is_scanned = (native_pages == 0 and total_pages > 0 and not is_url)
    is_mixed = (native_pages > 0 and ocr_pages > 0)

    assembled_chunks: List[str] = []
    chars_used = 0
    truncated = False

    for page_info in pages:
        method_label = page_info["method"].upper().replace("_", " ")
        if is_url:
            chunk = f"\n--- Webpage: {meta.get('filename', 'URL')} [Extraction: {method_label}] ---\n{page_info['text']}\n"
        elif is_image:
            chunk = f"\n--- Image [Extraction: {method_label}] ---\n{page_info['text']}\n"
        else:
            chunk = f"\n--- Page {page_info['page']} [Extraction: {method_label}] ---\n{page_info['text']}\n"

        if chars_used + len(chunk) > max_chars:
            truncated = True
            # If nothing has been included yet, or remaining budget allows,
            # include the header and as much page text as will fit into max_chars
            remaining_budget = max_chars - chars_used
            prefix = chunk.split("\n", 2)[0] + "\n" if "\n" in chunk else ""
            if remaining_budget > len(prefix) + 20:
                # Slice the chunk text to fit remaining budget
                sliced_chunk = chunk[:remaining_budget].rstrip() + "\n[Content truncated to fit context budget...]\n"
                assembled_chunks.append(sliced_chunk)
            break
        assembled_chunks.append(chunk)
        chars_used += len(chunk)

    context_str = "".join(assembled_chunks).strip()

    return {
        "context": context_str,
        "truncated": truncated,
        "pages_included": len(assembled_chunks),
        "total_pages": total_pages,
        "is_scanned": is_scanned,
        "is_mixed": is_mixed,
        "is_image": is_image,
        "is_url": is_url,
        "native_pages_count": native_pages,
        "ocr_pages_count": ocr_pages,
        "empty_pages_count": empty_pages,
        "ocr_available": is_tesseract_available(),
    }


def get_doc_metadata(doc_id: str) -> Optional[Dict[str, Any]]:
    """Returns stored metadata for a document ID, or attempts recovery from disk/database if backend restarted."""
    if doc_id in pdf_registry:
        return pdf_registry[doc_id]

    # Try database first
    try:
        from backend.storage import get_document_record
        db_doc = get_document_record(doc_id)
        if db_doc:
            storage_path = Path(db_doc["storage_path"])
            if storage_path.exists():
                suffix = storage_path.suffix.lower()
                is_img = suffix in {".png", ".jpg", ".jpeg", ".webp"}
                is_url_doc = suffix == ".txt" or db_doc.get("filename", "").startswith("http")
                file_kind = "url" if is_url_doc else ("image" if is_img else "pdf")
                meta = {
                    "doc_id": doc_id,
                    "filename": db_doc["filename"],
                    "page_count": db_doc["page_count"],
                    "size_bytes": db_doc["size_bytes"],
                    "sha256_prefix": db_doc["sha256_prefix"],
                    "uploaded_at": db_doc.get("created_at", ""),
                    "storage_path": str(storage_path),
                    "file_type": file_kind,
                    "ocr_available": is_tesseract_available(),
                }
                pdf_registry[doc_id] = meta
                return meta
    except Exception:
        pass

    # Try resolving path on disk
    try:
        storage_path = _find_storage_path(doc_id)
        if storage_path.exists():
            suffix = storage_path.suffix.lower()
            is_img = suffix in {".png", ".jpg", ".jpeg", ".webp"}
            is_url_doc = suffix == ".txt"
            file_kind = "url" if is_url_doc else ("image" if is_img else "pdf")
            page_count = 1
            if not is_img and not is_url_doc:
                try:
                    reader = PdfReader(str(storage_path))
                    page_count = len(reader.pages)
                except Exception:
                    page_count = 1
            meta = {
                "doc_id": doc_id,
                "filename": storage_path.name,
                "page_count": page_count,
                "size_bytes": storage_path.stat().st_size,
                "sha256_prefix": "",
                "uploaded_at": "",
                "storage_path": str(storage_path),
                "file_type": file_kind,
                "ocr_available": is_tesseract_available(),
            }
            pdf_registry[doc_id] = meta
            return meta
    except Exception:
        pass

    return None
