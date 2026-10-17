"""Environment-based configuration with startup validation.

Values come from process environment variables, optionally pre-loaded from a
local ``.env`` file (never committed). Nothing deployment-specific is hardcoded.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import quote, urlsplit

from .redact import redact_url, register_secret

CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
HIKVISION_STREAMS = {"main": "01", "sub": "02"}


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid. Messages never contain secrets."""


def build_hikvision_rtsp_url(
    host: str,
    username: str,
    password: str,
    channel: int = 1,
    stream: str = "sub",
    port: int = 554,
) -> str:
    """Build a Hikvision NVR RTSP URL: rtsp://user:pass@host:port/Streaming/Channels/<ch><01|02>.

    Channel 1 sub stream -> 102, channel 1 main stream -> 101. Credentials are
    percent-encoded so passwords containing '@', ':' or '/' still parse.
    """
    if stream not in HIKVISION_STREAMS:
        raise ConfigError(f"NVR_STREAM must be one of {sorted(HIKVISION_STREAMS)}, got {stream!r}")
    if not 1 <= channel <= 64:
        raise ConfigError(f"NVR_CHANNEL must be between 1 and 64, got {channel}")
    if not 1 <= port <= 65535:
        raise ConfigError(f"NVR_RTSP_PORT must be between 1 and 65535, got {port}")
    if not host or any(c in host for c in "/@ "):
        raise ConfigError("NVR_HOST must be a bare hostname or IP address")
    userinfo = f"{quote(username, safe='')}:{quote(password, safe='')}@" if username else ""
    host_part = f"[{host}]" if ":" in host else host
    return f"rtsp://{userinfo}{host_part}:{port}/Streaming/Channels/{channel}{HIKVISION_STREAMS[stream]}"


@dataclass(frozen=True)
class Settings:
    camera_id: str
    source: str  # RTSP URL or local video file path (dev/test only)
    source_kind: str  # "rtsp" or "file"
    frame_sample_fps: float = 2.0
    model_path: str = "yolo11m.pt"
    tracker_config: str = "bytetrack.yaml"
    detection_confidence: float = 0.35
    inference_image_size: int = 640
    device: str = ""  # "" lets Ultralytics choose; "cpu", "0" for first CUDA GPU
    # stream reliability
    open_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 10.0
    stall_timeout_seconds: float = 15.0
    reconnect_initial_seconds: float = 1.0
    reconnect_max_seconds: float = 60.0
    auth_failure_backoff_seconds: float = 300.0
    loop_file_source: bool = False
    # events
    worker_url: str = ""
    worker_ingest_token: str = field(default="", repr=False)
    event_publish_interval_seconds: float = 30.0
    event_min_interval_seconds: float = 2.0
    count_stable_samples: int = 2
    publish_queue_size: int = 100
    http_timeout_seconds: float = 10.0
    http_max_retries: int = 4
    http_backoff_initial_seconds: float = 1.0
    http_backoff_max_seconds: float = 30.0
    # local output
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 5
    event_log_enabled: bool = True
    status_file: str = "logs/status.json"
    status_interval_seconds: float = 5.0

    @property
    def redacted_source(self) -> str:
        return redact_url(self.source) if self.source_kind == "rtsp" else self.source

    @property
    def publishing_enabled(self) -> bool:
        return bool(self.worker_url)


def load_dotenv_if_present(path: str | os.PathLike = ".env") -> bool:
    """Load a local .env without overriding real environment variables."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is in requirements-edge.txt
        return False
    return load_dotenv(p, override=False)


def _get(env: Mapping[str, str], key: str, default: str = "") -> str:
    return (env.get(key) or default).strip()


def _float(env, key, default, lo=None, hi=None) -> float:
    raw = _get(env, key, str(default))
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from None
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ConfigError(f"{key} must be between {lo} and {hi}, got {value}")
    return value


def _int(env, key, default, lo=None, hi=None) -> int:
    raw = _get(env, key, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from None
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ConfigError(f"{key} must be between {lo} and {hi}, got {value}")
    return value


def _bool(env, key, default: bool) -> bool:
    raw = _get(env, key, "true" if default else "false").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{key} must be true/false, got {raw!r}")


def _resolve_source(env: Mapping[str, str]) -> tuple[str, str]:
    rtsp_url = _get(env, "RTSP_URL")
    nvr_host = _get(env, "NVR_HOST")
    video_file = _get(env, "VIDEO_FILE")
    chosen = [name for name, v in (("RTSP_URL", rtsp_url), ("NVR_HOST", nvr_host), ("VIDEO_FILE", video_file)) if v]
    if len(chosen) != 1:
        raise ConfigError(
            "Configure exactly one video source: RTSP_URL, or NVR_HOST (+ NVR_USERNAME/NVR_PASSWORD), "
            f"or VIDEO_FILE for local testing. Currently set: {chosen or 'none'}"
        )
    if rtsp_url:
        parts = urlsplit(rtsp_url)
        if parts.scheme.lower() not in {"rtsp", "rtsps"} or not parts.hostname:
            raise ConfigError(f"RTSP_URL must look like rtsp://host:port/path, got {redact_url(rtsp_url)!r}")
        register_secret(parts.password)
        return rtsp_url, "rtsp"
    if nvr_host:
        password = _get(env, "NVR_PASSWORD")
        username = _get(env, "NVR_USERNAME")
        if not username or not password:
            raise ConfigError("NVR_USERNAME and NVR_PASSWORD are required when NVR_HOST is set")
        register_secret(password)
        url = build_hikvision_rtsp_url(
            host=nvr_host,
            username=username,
            password=password,
            channel=_int(env, "NVR_CHANNEL", 1, 1, 64),
            stream=_get(env, "NVR_STREAM", "sub").lower(),
            port=_int(env, "NVR_RTSP_PORT", 554, 1, 65535),
        )
        return url, "rtsp"
    if not Path(video_file).is_file():
        raise ConfigError(f"VIDEO_FILE does not exist: {video_file!r}")
    return video_file, "file"


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Read and validate settings. Raises ConfigError with a secret-free message."""
    env = os.environ if env is None else env
    camera_id = _get(env, "CAMERA_ID")
    if not CAMERA_ID_RE.match(camera_id):
        raise ConfigError("CAMERA_ID is required: 1-64 chars of letters, digits, '-' or '_' (e.g. camera-01)")

    source, source_kind = _resolve_source(env)

    worker_url = _get(env, "WORKER_URL")
    token = _get(env, "WORKER_INGEST_TOKEN")
    if worker_url:
        parts = urlsplit(worker_url)
        local = parts.hostname in {"localhost", "127.0.0.1", "::1"}
        if parts.scheme != "https" and not (parts.scheme == "http" and local):
            raise ConfigError("WORKER_URL must use https:// (http:// is only allowed for localhost)")
        if parts.username or parts.password:
            raise ConfigError("WORKER_URL must not contain credentials; use WORKER_INGEST_TOKEN")
        if len(token) < 32:
            raise ConfigError("WORKER_INGEST_TOKEN must be set (at least 32 characters) when WORKER_URL is set")
    register_secret(token)

    settings = Settings(
        camera_id=camera_id,
        source=source,
        source_kind=source_kind,
        frame_sample_fps=_float(env, "FRAME_SAMPLE_FPS", 2.0, 0.1, 30.0),
        model_path=_get(env, "MODEL_PATH", "yolo11m.pt"),
        tracker_config=_get(env, "TRACKER_CONFIG", "bytetrack.yaml"),
        detection_confidence=_float(env, "DETECTION_CONFIDENCE", 0.35, 0.01, 0.99),
        inference_image_size=_int(env, "INFERENCE_IMAGE_SIZE", 640, 160, 1920),
        device=_get(env, "INFERENCE_DEVICE"),
        open_timeout_seconds=_float(env, "RTSP_OPEN_TIMEOUT_SECONDS", 10.0, 1.0, 120.0),
        read_timeout_seconds=_float(env, "RTSP_READ_TIMEOUT_SECONDS", 10.0, 1.0, 120.0),
        stall_timeout_seconds=_float(env, "STREAM_STALL_TIMEOUT_SECONDS", 15.0, 2.0, 600.0),
        reconnect_initial_seconds=_float(env, "RECONNECT_INITIAL_SECONDS", 1.0, 0.1, 60.0),
        reconnect_max_seconds=_float(env, "RECONNECT_MAX_SECONDS", 60.0, 1.0, 3600.0),
        auth_failure_backoff_seconds=_float(env, "AUTH_FAILURE_BACKOFF_SECONDS", 300.0, 10.0, 7200.0),
        loop_file_source=_bool(env, "LOOP_VIDEO_FILE", False),
        worker_url=worker_url,
        worker_ingest_token=token,
        event_publish_interval_seconds=_float(env, "EVENT_PUBLISH_INTERVAL_SECONDS", 30.0, 5.0, 3600.0),
        event_min_interval_seconds=_float(env, "EVENT_MIN_INTERVAL_SECONDS", 2.0, 0.5, 600.0),
        count_stable_samples=_int(env, "COUNT_STABLE_SAMPLES", 2, 1, 50),
        publish_queue_size=_int(env, "PUBLISH_QUEUE_SIZE", 100, 1, 10_000),
        http_timeout_seconds=_float(env, "HTTP_TIMEOUT_SECONDS", 10.0, 1.0, 60.0),
        http_max_retries=_int(env, "HTTP_MAX_RETRIES", 4, 0, 10),
        http_backoff_initial_seconds=_float(env, "HTTP_BACKOFF_INITIAL_SECONDS", 1.0, 0.1, 60.0),
        http_backoff_max_seconds=_float(env, "HTTP_BACKOFF_MAX_SECONDS", 30.0, 1.0, 600.0),
        log_dir=_get(env, "LOG_DIR", "logs"),
        log_level=_get(env, "LOG_LEVEL", "INFO").upper(),
        log_max_bytes=_int(env, "LOG_MAX_BYTES", 5_000_000, 10_000, 1_000_000_000),
        log_backup_count=_int(env, "LOG_BACKUP_COUNT", 5, 1, 100),
        event_log_enabled=_bool(env, "EVENT_LOG_ENABLED", True),
        status_file=_get(env, "STATUS_FILE", str(Path(_get(env, "LOG_DIR", "logs")) / "status.json")),
        status_interval_seconds=_float(env, "STATUS_INTERVAL_SECONDS", 5.0, 1.0, 300.0),
    )
    if settings.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError(f"LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR, got {settings.log_level!r}")
    if settings.reconnect_max_seconds < settings.reconnect_initial_seconds:
        raise ConfigError("RECONNECT_MAX_SECONDS must be >= RECONNECT_INITIAL_SECONDS")
    if settings.event_publish_interval_seconds < settings.event_min_interval_seconds:
        raise ConfigError("EVENT_PUBLISH_INTERVAL_SECONDS must be >= EVENT_MIN_INTERVAL_SECONDS")
    if not Path(settings.model_path).is_file():
        raise ConfigError(f"MODEL_PATH not found: {settings.model_path!r}")
    return settings
