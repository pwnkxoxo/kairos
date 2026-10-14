"""
KAIROS URL Service
Provides safe, SSRF-resistant URL ingestion, connection pinning, streaming fetch,
and HTML text extraction for Project KAIROS.

Security Principles:
  1. Connection Pinning (Anti-DNS-Rebinding / Anti-TOCTOU):
     Resolves DNS once and validates every returned IP address.
     Establishes the underlying TCP socket strictly to the pre-validated public IP address
     via custom `httpcore.NetworkBackend.connect_tcp`, while preserving the original
     target hostname for the HTTP `Host` header and TLS SNI / certificate validation.
  2. Strict Scheme & Port Allowlist:
     Only 'http' and 'https' schemes. Only standard ports (80 for http, 443 for https).
     Disallows userinfo credentials (e.g., http://user:pass@host).
  3. Comprehensive IP Range & Metadata Protection:
     Rejects loopback (127.0.0.0/8, ::1), private RFC1918 (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16),
     link-local (169.254.0.0/16, fe80::/10), reserved, multicast, unspecified (0.0.0.0, ::),
     IPv4-mapped IPv6 (::ffff:0:0/96), and cloud metadata services (169.254.169.254, metadata.google.internal).
     If ANY resolved IP is private or unsafe, the entire request is rejected immediately.
  4. Manual Redirect Verification:
     HTTP redirects are handled manually (max 3 hops) with zero automatic client redirection.
     Each hop undergoes the complete syntax and IP resolution/pinning validation.
     Cycles and cross-protocol leaks are rejected.
  5. Streaming & Resource Quotas:
     Max download limit: 2 MB (2,097,152 bytes). If content exceeds limit, stream is aborted immediately.
     Max extracted text context: 8,000 characters.
     Content-Type must be an acceptable HTML or plain text MIME type.
"""

from datetime import datetime, timezone
import html.parser
import ipaddress
import re
import socket
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
import uuid

import httpcore
import httpx

# ------------------------------------------------------------------------------
# 1. Constants & Limits
# ------------------------------------------------------------------------------
MAX_URL_FETCH_BYTES = 2 * 1024 * 1024  # 2 MB max download size
MAX_URL_REDIRECTS = 3                  # Max redirect hops
MAX_URL_CONTEXT_CHARS = 8_000          # Clamped context for AI analysis
URL_FETCH_TIMEOUT_SECONDS = 10.0       # Socket & request timeout

ALLOWED_URL_SCHEMES = {"http", "https"}
ALLOWED_URL_PORTS = {80, 443}

ALLOWED_MIME_PREFIXES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)

BLOCKED_HOSTNAMES = {
    "metadata.google.internal",
    "instance-data",
}


# ------------------------------------------------------------------------------
# 2. Custom Exceptions
# ------------------------------------------------------------------------------
class UrlServiceError(Exception):
    """Base exception for URL ingestion errors."""
    pass


class UrlValidationError(UrlServiceError):
    """Raised when URL syntax, scheme, port, or hostname is invalid or forbidden."""
    pass


class SsrfBlockedError(UrlServiceError):
    """Raised when a resolved IP address belongs to a private, loopback, or metadata network."""
    pass


class UrlFetchError(UrlServiceError):
    """Raised when a network failure, timeout, or HTTP protocol error occurs."""
    pass


class UrlContentTooLargeError(UrlServiceError):
    """Raised when the fetched content exceeds MAX_URL_FETCH_BYTES."""
    pass


class UrlUnsupportedContentTypeError(UrlServiceError):
    """Raised when the response Content-Type is not HTML or plain text."""
    pass


# ------------------------------------------------------------------------------
# 3. Connection Pinning Transport
# ------------------------------------------------------------------------------
class DeadlineStream(httpcore.NetworkStream):
    """
    Wraps an httpcore.NetworkStream to enforce an absolute monotonic deadline
    on all read, write, and TLS handshake operations, ensuring individual socket
    operations do not exceed the remaining wall-clock budget.
    """

    def __init__(self, stream: httpcore.NetworkStream, deadline: float):
        self._stream = stream
        self._deadline = deadline

    def read(self, max_bytes: int, timeout: Optional[float] = None) -> bytes:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise httpcore.ReadTimeout("Read operation timed out: fetch deadline exceeded.")
        effective_timeout = remaining if timeout is None else min(timeout, remaining)
        return self._stream.read(max_bytes, timeout=effective_timeout)

    def write(self, buffer: bytes, timeout: Optional[float] = None) -> None:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise httpcore.WriteTimeout("Write operation timed out: fetch deadline exceeded.")
        effective_timeout = remaining if timeout is None else min(timeout, remaining)
        return self._stream.write(buffer, timeout=effective_timeout)

    def close(self) -> None:
        self._stream.close()

    def start_tls(self, *args: Any, **kwargs: Any) -> httpcore.NetworkStream:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise httpcore.ConnectTimeout("TLS handshake timed out: fetch deadline exceeded.")
        if "timeout" in kwargs and kwargs["timeout"] is not None:
            kwargs["timeout"] = min(kwargs["timeout"], remaining)
        tls_stream = self._stream.start_tls(*args, **kwargs)
        return DeadlineStream(tls_stream, self._deadline)

    def get_extra_info(self, info: str) -> Any:
        return self._stream.get_extra_info(info)


class PinnedNetworkBackend(httpcore.SyncBackend):
    """
    Subclasses httpcore.SyncBackend to intercept connect_tcp and redirect
    the raw TCP socket connection strictly to a pre-validated IP address,
    preventing DNS rebinding / TOCTOU attacks while letting httpcore perform
    TLS SNI / certificate validation against the original server hostname.
    Optionally wraps the resulting stream in a DeadlineStream to enforce
    wall-clock deadlines across blocking socket operations.
    """

    def __init__(self, pinned_ip: str, deadline: Optional[float] = None):
        super().__init__()
        self.pinned_ip = pinned_ip
        self.deadline = deadline

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        local_address: Optional[str] = None,
        socket_options: Optional[Any] = None,
    ) -> httpcore.NetworkStream:
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise httpcore.ConnectTimeout("Connection timed out: fetch deadline exceeded.")
            effective_timeout = remaining if timeout is None else min(timeout, remaining)
        else:
            effective_timeout = timeout

        stream = super().connect_tcp(
            host=self.pinned_ip,
            port=port,
            timeout=effective_timeout,
            local_address=local_address,
            socket_options=socket_options,
        )
        if self.deadline is not None:
            return DeadlineStream(stream, self.deadline)
        return stream


class PinnedHTTPTransport(httpx.HTTPTransport):
    """
    httpx HTTPTransport configured with a PinnedNetworkBackend.
    """

    def __init__(self, pinned_ip: str, deadline: Optional[float] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.pinned_ip = pinned_ip
        self.deadline = deadline
        self._pool._network_backend = PinnedNetworkBackend(pinned_ip, deadline=deadline)


# ------------------------------------------------------------------------------
# 4. URL Validation & IP Filtering
# ------------------------------------------------------------------------------
def validate_url_syntax(url_str: str) -> Tuple[str, str, int, str]:
    """
    Validates URL syntax, scheme, credentials, and port.
    Returns (scheme, hostname, port, normalized_url).
    """
    if not url_str or not isinstance(url_str, str):
        raise UrlValidationError("URL must be a non-empty string.")

    cleaned_url = url_str.strip()
    if not cleaned_url:
        raise UrlValidationError("URL cannot be empty or whitespace.")

    try:
        parsed = urlparse(cleaned_url)
    except Exception as exc:
        raise UrlValidationError(f"Invalid URL syntax: {exc}")

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_URL_SCHEMES:
        raise UrlValidationError(
            f"Unsupported scheme '{scheme}'. Only HTTP and HTTPS are permitted."
        )

    raw_hostname = (parsed.hostname or "").lower()
    if not raw_hostname:
        raise UrlValidationError("URL must contain a valid hostname.")

    # Strip trailing dot consistently for canonical hostname comparison and DNS/TLS handling
    hostname = raw_hostname.rstrip(".")
    if not hostname:
        raise UrlValidationError("URL must contain a valid hostname.")

    if hostname in BLOCKED_HOSTNAMES:
        raise SsrfBlockedError(f"Access to hostname '{raw_hostname}' is blocked.")

    if parsed.username or parsed.password:
        raise UrlValidationError("URLs containing user credentials are not allowed.")

    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80

    if port not in ALLOWED_URL_PORTS:
        raise UrlValidationError(
            f"Port {port} is not allowed. Only standard ports (80, 443) are supported."
        )

    # Reconstruct normalized URL string without credentials or fragments, using canonical hostname
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    normalized = f"{scheme}://{hostname}{f':{port}' if parsed.port else ''}{path}{query}"
    return scheme, hostname, port, normalized


def is_ip_safe(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> Tuple[bool, str]:
    """
    Checks if an IP address is a safe, publicly routable address.
    Rejects loopback, private, link-local, reserved, multicast, unspecified,
    and cloud metadata service addresses.
    """
    if ip_obj.is_loopback:
        return False, "Loopback addresses are forbidden."
    if ip_obj.is_private:
        return False, "Private IP addresses are forbidden."
    if ip_obj.is_link_local:
        return False, "Link-local addresses are forbidden."
    if ip_obj.is_reserved:
        return False, "Reserved IP addresses are forbidden."
    if ip_obj.is_multicast:
        return False, "Multicast IP addresses are forbidden."
    if ip_obj.is_unspecified:
        return False, "Unspecified addresses (0.0.0.0 / ::) are forbidden."

    if isinstance(ip_obj, ipaddress.IPv6Address):
        # Disallow IPv4-mapped IPv6 (::ffff:x.x.x.x) if mapping to private or forbidden
        if ip_obj.ipv4_mapped:
            return is_ip_safe(ip_obj.ipv4_mapped)
        if ip_obj in ipaddress.IPv6Network("::ffff:0:0/96"):
            return False, "IPv4-mapped IPv6 addresses are forbidden."

    str_ip = str(ip_obj)
    if str_ip == "169.254.169.254":
        return False, "Cloud metadata IP address is forbidden."

    return True, "Safe"


def resolve_and_validate_host(hostname: str, port: int) -> str:
    """
    Resolves hostname to IP addresses using socket.getaddrinfo.
    Verifies that ALL resolved IP addresses are safe public IPs.
    Returns the first verified IP address as a string to pin for connection.
    """
    # Check if hostname itself is directly a numeric IP literal
    try:
        direct_ip = ipaddress.ip_address(hostname)
        safe, reason = is_ip_safe(direct_ip)
        if not safe:
            raise SsrfBlockedError(f"IP address '{hostname}' is not permitted: {reason}")
        return str(direct_ip)
    except ValueError:
        pass  # It is a domain name, proceed to DNS resolution

    try:
        addrinfo = socket.getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise UrlFetchError(f"Failed to resolve hostname '{hostname}': {exc}")

    if not addrinfo:
        raise UrlFetchError(f"No IP addresses resolved for hostname '{hostname}'.")

    resolved_ips: List[str] = []
    for entry in addrinfo:
        sockaddr = entry[4]
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            raise SsrfBlockedError(f"Invalid resolved IP format: '{ip_str}'")

        safe, reason = is_ip_safe(ip_obj)
        if not safe:
            raise SsrfBlockedError(
                f"Hostname '{hostname}' resolved to forbidden address '{ip_str}': {reason}"
            )
        resolved_ips.append(str(ip_obj))

    # Return the first safe resolved IP address to pin the connection
    return resolved_ips[0]


# ------------------------------------------------------------------------------
# 5. HTML Content Extractor
# ------------------------------------------------------------------------------
class SafeHTMLTextExtractor(html.parser.HTMLParser):
    """
    Lightweight, robust HTML parser using standard library html.parser.
    Extracts the document <title> and readable text while stripping:
      <script>, <style>, <noscript>, <svg>, <nav>, <header>, <footer>, <iframe>, <template>
    Normalizes whitespace and formats paragraph breaks.
    Tolerates malformed, mismatched, unclosed, or nested tags without leaking ignored text.
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._title_parts: List[str] = []
        self._text_parts: List[str] = []
        self._ignore_tags = {
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "header",
            "footer",
            "iframe",
            "template",
        }
        self._ignored_stack: List[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        lower_tag = tag.lower()
        if lower_tag == "title":
            self._in_title = True
        elif lower_tag in self._ignore_tags:
            self._ignored_stack.append(lower_tag)
        elif not self._ignored_stack:
            if lower_tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br", "article", "section"):
                self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag == "title":
            self._in_title = False
        elif self._ignored_stack:
            # If the closing tag matches any tag in the active ignored stack,
            # pop up to and including that tag to recover from nested or mismatched closures.
            if lower_tag in self._ignored_stack:
                idx = len(self._ignored_stack) - 1 - self._ignored_stack[::-1].index(lower_tag)
                self._ignored_stack = self._ignored_stack[:idx]
        else:
            if lower_tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "article", "section"):
                self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        elif not self._ignored_stack:
            self._text_parts.append(data)

    def get_result(self) -> Tuple[str, str]:
        title = " ".join("".join(self._title_parts).split()).strip()
        raw_text = "".join(self._text_parts)
        # Normalize whitespace per line while preserving meaningful linebreaks
        cleaned_lines: List[str] = []
        for line in raw_text.splitlines():
            normalized = " ".join(line.split())
            if normalized:
                cleaned_lines.append(normalized)
        cleaned_text = "\n".join(cleaned_lines)
        return title, cleaned_text


def extract_html_content(
    html_raw: str,
    source_url: str,
    max_chars: int = MAX_URL_CONTEXT_CHARS,
) -> Dict[str, Any]:
    """
    Parses HTML content, extracts title and readable text, strips noise,
    and returns bounded text suitable for Gemma model analysis.
    """
    parser = SafeHTMLTextExtractor()
    try:
        parser.feed(html_raw)
        title, text = parser.get_result()
    except Exception:
        # Fallback to basic regex stripping if parser hits malformed structure
        title = ""
        no_tags = re.sub(r"<[^>]+>", " ", html_raw)
        text = " ".join(no_tags.split()).strip()

    if not title:
        # Derive title from domain or path if <title> was missing
        parsed = urlparse(source_url)
        title = parsed.hostname or "Webpage Content"

    # Clamp content to max_chars
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
        truncated = True

    return {
        "title": title,
        "text": text,
        "source_url": source_url,
        "char_count": len(text),
        "truncated": truncated,
    }


# ------------------------------------------------------------------------------
# 6. Safe Fetching Pipeline
# ------------------------------------------------------------------------------
def safe_fetch_url(
    target_url: str,
    timeout_seconds: float = URL_FETCH_TIMEOUT_SECONDS,
    max_bytes: int = MAX_URL_FETCH_BYTES,
    max_redirects: int = MAX_URL_REDIRECTS,
) -> Tuple[str, str, str]:
    """
    Executes a safe HTTP GET request with connection pinning and bounded streaming.
    Follows redirects manually up to max_redirects hops.
    Validates DNS and pins socket connection on each hop.
    Returns (final_url, response_text, content_type).
    """
    current_url = target_url
    visited_urls: Set[str] = set()
    fetch_deadline = time.monotonic() + timeout_seconds

    headers = {
        "User-Agent": "KAIROS-Reader/1.0 (+https://github.com/pwnkxoxo/kairos)",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
        "Accept-Encoding": "gzip, deflate",
    }

    for hop in range(max_redirects + 1):
        remaining_timeout = fetch_deadline - time.monotonic()
        if remaining_timeout <= 0:
            raise UrlFetchError(f"Overall fetch deadline of {timeout_seconds}s exceeded.")

        scheme, hostname, port, normalized_url = validate_url_syntax(current_url)

        if normalized_url in visited_urls:
            raise UrlFetchError(f"Redirect loop detected: '{normalized_url}' was already visited.")
        visited_urls.add(normalized_url)

        # DNS resolution & SSRF validation
        pinned_ip = resolve_and_validate_host(hostname, port)

        # Re-check deadline after DNS resolution
        remaining_timeout = fetch_deadline - time.monotonic()
        if remaining_timeout <= 0:
            raise UrlFetchError(f"Overall fetch deadline of {timeout_seconds}s exceeded.")

        # Build connection-pinned transport with overall monotonic deadline
        transport = PinnedHTTPTransport(
            pinned_ip=pinned_ip,
            deadline=fetch_deadline,
            verify=True,
            http1=True,
            http2=False,
        )

        try:
            with httpx.Client(
                transport=transport,
                timeout=remaining_timeout,
                follow_redirects=False,
            ) as client:
                req = client.build_request("GET", normalized_url, headers=headers)
                resp = client.send(req, stream=True)

                # Check for redirects (301, 302, 303, 307, 308)
                if resp.status_code in (301, 302, 303, 307, 308):
                    resp.close()
                    location = resp.headers.get("Location")
                    if not location:
                        raise UrlFetchError(
                            f"Redirect status {resp.status_code} received without Location header."
                        )
                    if hop >= max_redirects:
                        raise UrlFetchError(
                            f"Exceeded maximum redirect limit of {max_redirects} hops."
                        )
                    # Resolve relative redirect against current_url
                    current_url = urljoin(normalized_url, location)
                    continue

                # Not a redirect - verify status code
                if resp.status_code >= 400:
                    resp.close()
                    raise UrlFetchError(
                        f"Server returned HTTP error status {resp.status_code} ({resp.reason_phrase})."
                    )

                # Validate Content-Type
                content_type = resp.headers.get("Content-Type", "").lower()
                is_allowed_mime = any(content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES)
                # If Content-Type is missing or generic text, allow fallback to parser
                if content_type and not is_allowed_mime and not content_type.startswith("text/"):
                    resp.close()
                    raise UrlUnsupportedContentTypeError(
                        f"Unsupported Content-Type '{content_type}'. Only HTML and plain text pages are supported."
                    )

                # Stream response bytes with hard ceiling and deadline checks
                content_chunks: List[bytes] = []
                total_bytes = 0

                try:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        if time.monotonic() > fetch_deadline:
                            raise UrlFetchError(
                                f"Overall fetch deadline of {timeout_seconds}s exceeded during streaming."
                            )
                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            raise UrlContentTooLargeError(
                                f"Webpage exceeds maximum size limit of {max_bytes // (1024 * 1024)} MB."
                            )
                        content_chunks.append(chunk)
                finally:
                    resp.close()

                raw_bytes = b"".join(content_chunks)

                # Decode text using response encoding or utf-8 fallback
                encoding = resp.encoding or "utf-8"
                try:
                    decoded_text = raw_bytes.decode(encoding, errors="replace")
                except Exception:
                    decoded_text = raw_bytes.decode("utf-8", errors="replace")

                return normalized_url, decoded_text, content_type

        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            raise UrlFetchError(f"Connection timed out while fetching '{hostname}': {exc}")
        except httpx.ConnectError as exc:
            raise UrlFetchError(f"Failed to connect to '{hostname}': {exc}")
        except UrlServiceError:
            raise
        except Exception as exc:
            raise UrlFetchError(f"Unexpected error fetching URL: {exc}")

    raise UrlFetchError(f"Exceeded maximum redirect limit of {max_redirects} hops.")
