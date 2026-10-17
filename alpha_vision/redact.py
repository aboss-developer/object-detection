"""Helpers that keep credentials out of logs, errors, health files and events."""

from __future__ import annotations

import logging
import re
import threading
from urllib.parse import urlsplit, urlunsplit

# scheme://user:pass@host  ->  scheme://***@host   (also matches user-only userinfo)
_URL_USERINFO = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<userinfo>[^/\s@]+)@")
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_TELEGRAM_BOT = re.compile(r"bot\d+:[A-Za-z0-9_-]+")

_registered_secrets: set[str] = set()
_lock = threading.Lock()


def register_secret(value: str | None) -> None:
    """Remember a secret so that any later log line containing it is masked."""
    if value and len(value) >= 4:
        with _lock:
            _registered_secrets.add(value)


def redact_url(url: str | None) -> str:
    """Return the URL with any userinfo (username/password) replaced by ***."""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return redact_text(url)
    if parts.username is None and parts.password is None:
        return url
    host = parts.hostname or ""
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"***@{host}" + (f":{port}" if port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def redact_text(text: str) -> str:
    """Mask URL credentials, bearer tokens, Telegram bot tokens and registered secrets."""
    if not text:
        return text
    out = _URL_USERINFO.sub(lambda m: f"{m.group('scheme')}***@", text)
    out = _BEARER.sub(r"\1***", out)
    out = _TELEGRAM_BOT.sub("bot***", out)
    with _lock:
        secrets = sorted(_registered_secrets, key=len, reverse=True)
    for secret in secrets:
        out = out.replace(secret, "***")
    return out


class RedactingFilter(logging.Filter):
    """Logging filter that redacts the final formatted message and exception text."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - malformed format args
            message = str(record.msg)
        record.msg = redact_text(message)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact_text(record.exc_text)
        return True
