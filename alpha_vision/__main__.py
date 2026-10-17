"""CLI entry point.

    python -m alpha_vision run              # headless service (default)
    python -m alpha_vision check-config     # validate configuration only
    python -m alpha_vision check-rtsp       # diagnose the configured RTSP stream and read a few frames
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import ConfigError, load_dotenv_if_present, load_settings
from .logging_setup import make_event_log, setup_logging
from .redact import redact_text

log = logging.getLogger("alpha_vision")


def _settings(args):
    if not args.no_dotenv:
        load_dotenv_if_present(args.env_file)
    return load_settings()


def cmd_check_config(args) -> int:
    s = _settings(args)
    print(f"OK camera_id={s.camera_id} input={s.source_kind} source={s.redacted_source} "
          f"sample_fps={s.frame_sample_fps} publishing={'on' if s.publishing_enabled else 'off (WORKER_URL unset)'}")
    return 0


def cmd_check_rtsp(args) -> int:
    from .rtsp_probe import probe_rtsp
    from .stream import open_opencv_capture

    s = _settings(args)
    if s.source_kind != "rtsp":
        print("check-rtsp needs RTSP_URL or NVR_HOST; VIDEO_FILE is configured instead.")
        return 2
    print(f"Probing {s.redacted_source} ...")
    result = probe_rtsp(s.source, timeout=args.timeout)
    status = f" (RTSP {result.status_code})" if result.status_code else ""
    print(f"[{'PASS' if result.ok else 'FAIL'}] RTSP handshake: {result.stage}{status}")
    if result.codecs:
        print(f"       codecs advertised: {', '.join(result.codecs)}")
    if not result.ok:
        print(f"       hint: {result.hint}")
        if result.detail:
            print(f"       detail: {redact_text(result.detail)}")
        return 1

    cap = open_opencv_capture(s.source, "rtsp", s.open_timeout_seconds, s.read_timeout_seconds)
    try:
        if not cap.isOpened():
            print("[FAIL] OpenCV/FFmpeg could not open the stream although RTSP DESCRIBE succeeded.\n"
                  "       hint: unsupported codec (try H.264 on the sub stream) or FFmpeg build issue.")
            return 1
        read, started = 0, time.monotonic()
        shape = None
        while read < args.frames and time.monotonic() - started < args.timeout * 3:
            ok, frame = cap.read()
            if not ok:
                break
            read += 1
            shape = frame.shape
        elapsed = time.monotonic() - started
        if read == 0:
            print("[FAIL] Connected but no frames decoded. hint: check codec / bandwidth; try NVR_STREAM=sub.")
            return 1
        print(f"[PASS] Decoded {read} frames in {elapsed:.1f}s, frame size {shape[1]}x{shape[0]}")
        return 0
    finally:
        cap.release()


def cmd_run(args) -> int:
    from .service import build_service

    s = _settings(args)
    setup_logging(s.log_dir, s.log_level, s.log_max_bytes, s.log_backup_count)
    event_log = make_event_log(s.log_dir, s.log_max_bytes, s.log_backup_count) if s.event_log_enabled else None
    service = build_service(s, event_log)
    service.install_signal_handlers()
    return service.run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="alpha_vision", description="Alpha Vision single-camera edge service")
    parser.add_argument("--env-file", default=".env", help="optional dotenv file (default: .env)")
    parser.add_argument("--no-dotenv", action="store_true", help="ignore .env; use process environment only")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="run the headless camera service")
    sub.add_parser("check-config", help="validate configuration and exit")
    rtsp = sub.add_parser("check-rtsp", help="test the configured RTSP stream")
    rtsp.add_argument("--timeout", type=float, default=5.0)
    rtsp.add_argument("--frames", type=int, default=10)
    args = parser.parse_args(argv)
    command = args.command or "run"
    try:
        return {"run": cmd_run, "check-config": cmd_check_config, "check-rtsp": cmd_check_rtsp}[command](args)
    except ConfigError as exc:
        print(f"Configuration error: {redact_text(str(exc))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
