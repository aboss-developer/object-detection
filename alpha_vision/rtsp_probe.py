"""Minimal RTSP DESCRIBE probe used to diagnose connection failures precisely.

OpenCV only reports "could not open"; this probe separates DNS, TCP/port, auth
(401), wrong channel/stream (404) and codec issues. It sends at most one
authenticated request per call, because Hikvision NVRs lock out an IP address
after repeated failed logins.
"""

from __future__ import annotations

import hashlib
import os
import re
import socket
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit, urlunsplit

from .redact import redact_url

HINTS = {
    "invalid_url": "RTSP URL is malformed. Expected rtsp://<user>:<pass>@<nvr-ip>:554/Streaming/Channels/102",
    "dns": "Host name did not resolve. Use the NVR's IP address or fix DNS.",
    "tcp_refused": "Host reachable but the RTSP port refused the connection. Check NVR_RTSP_PORT (default 554) "
                   "and that RTSP is enabled on the NVR (Configuration > Network > Advanced > Ports).",
    "tcp_timeout": "No answer on the RTSP port. Check the IP, that this machine is on the NVR's network/VLAN, "
                   "and any firewall between them.",
    "auth_failed": "NVR rejected the username/password. Verify credentials in the NVR web UI. Repeated failures "
                   "can lock this IP out for ~30 minutes, so the service backs off for AUTH_FAILURE_BACKOFF_SECONDS.",
    "not_found": "Stream path not found. Check NVR_CHANNEL / NVR_STREAM (channel 1 sub = 102, main = 101) and that "
                 "the camera is online on that NVR channel.",
    "forbidden": "Access denied. The NVR user may lack 'Remote Live View' permission for this channel.",
    "protocol": "Unexpected RTSP response. Check that the port is the RTSP port, not HTTP (80) or SDK (8000).",
    "ok": "RTSP DESCRIBE succeeded.",
}


@dataclass
class ProbeResult:
    ok: bool
    stage: str  # key into HINTS
    status_code: int | None = None
    codecs: tuple[str, ...] = ()
    detail: str = ""

    @property
    def hint(self) -> str:
        return HINTS.get(self.stage, "")


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()  # noqa: S324 - mandated by RFC 2617 digest auth


def _parse_auth_params(header: str) -> dict[str, str]:
    return {k.lower(): v1 or v2 for k, v1, v2 in re.findall(r'(\w+)=(?:"([^"]*)"|([^,\s]*))', header)}


def build_authorization(method: str, uri: str, username: str, password: str, challenges: list[str]) -> str | None:
    """Build an Authorization header for the strongest supported challenge (Digest > Basic)."""
    digest = next((c for c in challenges if c.lower().startswith("digest")), None)
    if digest:
        p = _parse_auth_params(digest)
        realm, nonce = p.get("realm", ""), p.get("nonce", "")
        ha1 = _md5(f"{username}:{realm}:{password}")
        ha2 = _md5(f"{method}:{uri}")
        qop = p.get("qop", "")
        if "auth" in [q.strip() for q in qop.split(",")]:
            nc, cnonce = "00000001", os.urandom(8).hex()
            response = _md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")
            extra = f', qop=auth, nc={nc}, cnonce="{cnonce}"'
        else:
            response = _md5(f"{ha1}:{nonce}:{ha2}")
            extra = ""
        opaque = f', opaque="{p["opaque"]}"' if "opaque" in p else ""
        return (f'Digest username="{username}", realm="{realm}", nonce="{nonce}", uri="{uri}", '
                f'response="{response}"{opaque}{extra}')
    if any(c.lower().startswith("basic") for c in challenges):
        import base64

        return "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
    return None


def _read_response(sock: socket.socket, limit: int = 65536) -> tuple[int, dict[str, list[str]], str]:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
        if len(data) > limit:
            break
    head, _, rest = data.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    match = re.match(r"RTSP/1\.\d (\d{3})", lines[0] if lines else "")
    if not match:
        raise ValueError("not an RTSP response")
    headers: dict[str, list[str]] = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers.setdefault(k.strip().lower(), []).append(v.strip())
    length = int(headers.get("content-length", ["0"])[0] or 0)
    while len(rest) < length:
        chunk = sock.recv(4096)
        if not chunk:
            break
        rest += chunk
    return int(match.group(1)), headers, rest[:length].decode("latin-1", "replace")


def _codecs(sdp: str) -> tuple[str, ...]:
    return tuple(sorted({m.upper() for m in re.findall(r"a=rtpmap:\d+ ([A-Za-z0-9-]+)/", sdp)}))


def probe_rtsp(url: str, timeout: float = 5.0) -> ProbeResult:
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port or 554
    except ValueError as exc:
        return ProbeResult(False, "invalid_url", detail=str(exc))
    if parts.scheme.lower() != "rtsp" or not host:
        return ProbeResult(False, "invalid_url", detail=redact_url(url))
    username = unquote(parts.username or "")
    password = unquote(parts.password or "")
    netloc = (f"[{host}]" if ":" in host else host) + f":{port}"
    clean_uri = urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return ProbeResult(False, "dns", detail=str(exc))
    try:
        sock = socket.create_connection(infos[0][4][:2], timeout=timeout)
    except ConnectionRefusedError as exc:
        return ProbeResult(False, "tcp_refused", detail=str(exc))
    except (socket.timeout, TimeoutError) as exc:
        return ProbeResult(False, "tcp_timeout", detail=str(exc) or "timed out")
    except OSError as exc:
        return ProbeResult(False, "tcp_timeout", detail=str(exc))

    def describe(cseq: int, auth: str | None) -> tuple[int, dict[str, list[str]], str]:
        req = (f"DESCRIBE {clean_uri} RTSP/1.0\r\nCSeq: {cseq}\r\nAccept: application/sdp\r\n"
               f"User-Agent: alpha-vision-probe\r\n" + (f"Authorization: {auth}\r\n" if auth else "") + "\r\n")
        sock.sendall(req.encode("latin-1"))
        return _read_response(sock)

    try:
        with sock:
            sock.settimeout(timeout)
            status, headers, body = describe(1, None)
            if status == 401:
                if not username:
                    return ProbeResult(False, "auth_failed", 401, detail="credentials required but none given")
                auth = build_authorization("DESCRIBE", clean_uri, username, password,
                                           headers.get("www-authenticate", []))
                if auth is None:
                    return ProbeResult(False, "protocol", 401, detail="unsupported auth scheme")
                status, headers, body = describe(2, auth)
    except (socket.timeout, TimeoutError):
        return ProbeResult(False, "tcp_timeout", detail="no RTSP response before timeout")
    except (OSError, ValueError) as exc:
        return ProbeResult(False, "protocol", detail=type(exc).__name__)

    if status == 200:
        return ProbeResult(True, "ok", 200, codecs=_codecs(body))
    stage = {401: "auth_failed", 403: "forbidden", 404: "not_found"}.get(status, "protocol")
    return ProbeResult(False, stage, status)
