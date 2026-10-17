import logging

import pytest

from alpha_vision.config import ConfigError, build_hikvision_rtsp_url, load_settings
from alpha_vision.logging_setup import JsonFormatter
from alpha_vision.redact import RedactingFilter, redact_text, redact_url, register_secret

TOKEN = "t" * 40


def test_hikvision_url_sub_and_main_streams():
    assert build_hikvision_rtsp_url("192.0.2.10", "viewer", "pw", 1, "sub") == \
        "rtsp://viewer:pw@192.0.2.10:554/Streaming/Channels/102"
    assert build_hikvision_rtsp_url("192.0.2.10", "viewer", "pw", 1, "main", 8554) == \
        "rtsp://viewer:pw@192.0.2.10:8554/Streaming/Channels/101"
    assert build_hikvision_rtsp_url("nvr", "u", "p", 8, "sub").endswith("/Streaming/Channels/802")


def test_hikvision_url_encodes_special_characters():
    url = build_hikvision_rtsp_url("192.0.2.10", "admin", "p@ss:w/rd", 1, "sub")
    assert "p%40ss%3Aw%2Frd@" in url
    assert redact_url(url) == "rtsp://***@192.0.2.10:554/Streaming/Channels/102"


@pytest.mark.parametrize("kwargs", [dict(stream="third"), dict(channel=0), dict(port=70000), dict(host="a/b")])
def test_hikvision_url_rejects_bad_values(kwargs):
    args = dict(host="192.0.2.10", username="u", password="p", channel=1, stream="sub", port=554) | kwargs
    with pytest.raises(ConfigError):
        build_hikvision_rtsp_url(**args)


def test_settings_from_nvr_components(base_env):
    env = dict(base_env)
    del env["VIDEO_FILE"]
    env.update(NVR_HOST="192.0.2.10", NVR_USERNAME="viewer", NVR_PASSWORD="S3cret!pw", NVR_CHANNEL="1",
               NVR_STREAM="sub")
    s = load_settings(env)
    assert s.source_kind == "rtsp"
    assert s.source.endswith("/Streaming/Channels/102")
    assert "S3cret" not in s.redacted_source
    assert "S3cret" not in repr(s.redacted_source)


def test_exactly_one_source_required(base_env):
    env = dict(base_env, RTSP_URL="rtsp://u:p@192.0.2.10:554/Streaming/Channels/102")
    with pytest.raises(ConfigError, match="exactly one"):
        load_settings(env)
    env = {k: v for k, v in base_env.items() if k != "VIDEO_FILE"}
    with pytest.raises(ConfigError, match="exactly one"):
        load_settings(env)


def test_config_error_never_contains_password(base_env):
    env = {k: v for k, v in base_env.items() if k != "VIDEO_FILE"}
    env["RTSP_URL"] = "http://admin:TopSecret99@192.0.2.10/stream"
    with pytest.raises(ConfigError) as exc:
        load_settings(env)
    assert "TopSecret99" not in str(exc.value)


@pytest.mark.parametrize("key,value", [("FRAME_SAMPLE_FPS", "0"), ("FRAME_SAMPLE_FPS", "abc"),
                                       ("CAMERA_ID", "bad id!"), ("CAMERA_ID", ""), ("LOG_LEVEL", "LOUD"),
                                       ("EVENT_PUBLISH_INTERVAL_SECONDS", "1")])
def test_invalid_values_rejected(base_env, key, value):
    with pytest.raises(ConfigError):
        load_settings(dict(base_env, **{key: value}))


def test_worker_settings_validation(base_env):
    with pytest.raises(ConfigError, match="https"):
        load_settings(dict(base_env, WORKER_URL="http://example.com/ingest", WORKER_INGEST_TOKEN=TOKEN))
    with pytest.raises(ConfigError, match="WORKER_INGEST_TOKEN"):
        load_settings(dict(base_env, WORKER_URL="https://example.com/ingest", WORKER_INGEST_TOKEN="short"))
    s = load_settings(dict(base_env, WORKER_URL="http://127.0.0.1:8787/v1/events", WORKER_INGEST_TOKEN=TOKEN))
    assert s.publishing_enabled
    assert TOKEN not in repr(s)


def test_defaults(base_env):
    s = load_settings(base_env)
    assert s.frame_sample_fps == 2.0
    assert s.source_kind == "file"
    assert not s.publishing_enabled


def test_redact_text_patterns():
    text = "open rtsp://admin:pw123@10.0.0.5:554/x failed; Authorization: Bearer abc.def-123; bot123456:AAbbCC_dd"
    out = redact_text(text)
    assert "pw123" not in out and "abc.def-123" not in out and "AAbbCC_dd" not in out
    assert "rtsp://***@10.0.0.5:554/x" in out


def test_logging_filter_redacts_args_and_exceptions():
    register_secret("hunter2-registered")
    record = logging.LogRecord("t", logging.ERROR, __file__, 1, "connect %s", ("rtsp://u:hunter2@h/x",), None)
    try:
        raise RuntimeError("token hunter2-registered leaked")
    except RuntimeError:
        import sys

        record.exc_info = sys.exc_info()
    RedactingFilter().filter(record)
    line = JsonFormatter().format(record)
    assert "hunter2" not in line
    assert "***" in line
